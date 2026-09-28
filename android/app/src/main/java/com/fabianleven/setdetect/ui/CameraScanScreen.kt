package com.fabianleven.setdetect.ui

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.view.Surface
import androidx.activity.compose.BackHandler
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.graphics.createBitmap
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.DetectedCard
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.NativeSetDetector
import com.fabianleven.setdetect.ui.components.ScanReviewOverlay
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

// Duration of the post-tap review pause: cards stay on screen, emphasized
// and dimmed, before auto-proceeding, giving time to catch a bad scan and
// back out before it's submitted.
private const val ScanReviewDurationMillis = 3000

// How long the review's dim/outline entrance animation takes to fade and
// pop in once it starts.
private const val ScanReviewEntranceMillis = 220

// Starting scale for each card's highlight ring, animated up to 1f with a
// spring when review starts, for a pop on entry.
private const val ScanReviewPopStartScale = 0.75f

private enum class ScanPhase {
    /** Live camera feed and detection running; Scan available. */
    Live,

    /** Paused on a frozen snapshot; cancellable via back or any tap. */
    Reviewing,

    /**
     * The review pause elapsed without cancellation: onCardsDetected has
     * been called and navigation away is in flight. The live preview stays
     * hidden and nothing here is interactive again in this phase.
     */
    Committed
}

/** What's being reviewed, captured in one shot when a review pause starts. */
private data class ScanSnapshot(
    val detection: Detection?,
    val cards: List<DetectedCard>,
    val matrix: Matrix?,
    val bitmap: Bitmap?
)

/**
 * Hands off the one analysis-bitmap frame the review pause is currently
 * showing, if any, so the analyzer thread never recycles a bitmap the UI
 * thread is displaying.
 *
 * [record] and [take] are the only points of contact between the two
 * threads and are synchronized on the same lock, so whichever runs second
 * always sees the other's completed effect, never a half-applied one.
 */
private class LatestBitmapHolder {
    private val lock = Any()
    private var bitmap: Bitmap? = null

    /**
     * Records this frame's bitmap as the latest live one, recycling
     * whatever it supersedes, unless [protectedBitmap] says that one is
     * currently held for review, in which case it must survive until
     * [release] releases it.
     *
     * Both possible lock-acquisition orders relative to [take] are safe:
     * if this runs first, [take] simply reads the bitmap recorded here; if
     * [take] runs first (claiming the previous bitmap for review before
     * this call decides what to recycle), [protectedBitmap] reflects that
     * claim and this skips recycling it.
     */
    fun record(newBitmap: Bitmap, protectedBitmap: () -> Bitmap?) {
        synchronized(lock) {
            val previous = bitmap
            if (previous !== protectedBitmap()) previous?.recycle()
            bitmap = newBitmap
        }
    }

    /** Claims the current bitmap, if any, for the caller to hold onto. */
    fun take(): Bitmap? = synchronized(lock) { bitmap }

    fun release() {
        synchronized(lock) {
            bitmap?.recycle()
            bitmap = null
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
fun CameraScanScreen(
    onCardsDetected: (Detection) -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var detector by remember { mutableStateOf<NativeSetDetector?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var currentDetection by remember { mutableStateOf<Detection?>(null) }
    var detectedCards by remember { mutableStateOf<List<DetectedCard>>(emptyList()) }
    val loadingText = stringResource(R.string.camera_loading)
    val readyText = stringResource(R.string.camera_ready)
    val detectorInitFailedText = stringResource(R.string.camera_detector_init_failed)
    val cameraStartFailedText = stringResource(R.string.camera_start_failed)

    var initProgress by remember { mutableStateOf(loadingText) }
    var analysisToViewMatrix by remember { mutableStateOf<Matrix?>(null) }

    var scanPhase by remember { mutableStateOf(ScanPhase.Live) }
    var reviewSnapshot by remember { mutableStateOf<ScanSnapshot?>(null) }
    // 1f = review pause just started, 0f = elapsed; drives both the timer
    // bar and the auto-proceed itself, so the two can never drift apart.
    val reviewCountdown = remember { Animatable(1f) }
    // Current scale of the review's highlight-ring pop-in animation.
    val reviewPopScale = remember { Animatable(ScanReviewPopStartScale) }

    val scope = rememberCoroutineScope()
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }
    val latestBitmap = remember { LatestBitmapHolder() }

    // Cancels the review pause and returns to the live view; there's
    // nothing to navigate away from yet since onCardsDetected hasn't been
    // called. Triggered by system back and by tapping anywhere during the
    // pause.
    val cancelReview: () -> Unit = {
        scanPhase = ScanPhase.Live
        reviewSnapshot?.bitmap?.recycle()
        reviewSnapshot = null
    }
    BackHandler(enabled = scanPhase == ScanPhase.Reviewing) { cancelReview() }

    // Completes the review, whether the countdown ran out or the checkmark
    // was tapped early. Setting scanPhase to Committed before calling
    // onCardsDetected cancels the countdown coroutine, since its
    // LaunchedEffect is keyed on scanPhase; that matters when this runs
    // early, so the countdown can't also call onCardsDetected once it
    // separately elapses.
    val commitReview: () -> Unit = {
        scanPhase = ScanPhase.Committed
        reviewSnapshot?.detection?.let { onCardsDetected(it) }
    }

    // Freezes what's currently detected and enters the review pause. Shared
    // by the Scan button and by tapping anywhere on the live view below, so
    // there's a single place that decides what "start reviewing" means.
    val startReview: () -> Unit = {
        reviewSnapshot = ScanSnapshot(
            detection = currentDetection,
            cards = detectedCards,
            matrix = analysisToViewMatrix,
            bitmap = latestBitmap.take()
        )
        scanPhase = ScanPhase.Reviewing
    }

    LaunchedEffect(scanPhase) {
        if (scanPhase == ScanPhase.Reviewing) {
            reviewCountdown.snapTo(1f)
            reviewPopScale.snapTo(ScanReviewPopStartScale)
            launch {
                reviewPopScale.animateTo(
                    1f,
                    spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessMedium)
                )
            }
            reviewCountdown.animateTo(0f, tween(ScanReviewDurationMillis, easing = LinearEasing))
            // Reached only if uninterrupted: cancelling or committing early
            // both change scanPhase, which cancels this coroutine via
            // structured concurrency before this point.
            commitReview()
        }
    }

    LaunchedEffect(Unit) {
        scope.launch(Dispatchers.IO) {
            try {
                val cornerModel = copyAssetToFile(context, "models/card_corners.onnx")
                val classModel = copyAssetToFile(context, "models/card_classifier_quant.onnx")
                detector = NativeSetDetector(cornerModel.absolutePath, classModel.absolutePath)
                initProgress = readyText
            } catch (e: Exception) {
                error = "$detectorInitFailedText: ${e.message}"
                Log.e("CameraScanScreen", "Detector init failed", e)
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            detector?.close()
            analyzerExecutor.shutdown()
            latestBitmap.release()
            reviewSnapshot?.bitmap?.recycle()
        }
    }

    if (error != null) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(error!!, color = MaterialTheme.colorScheme.error)
        }
    } else if (cameraPermissionState.status.isGranted) {
        Box(modifier = modifier.fillMaxSize()) {
            if (detector == null) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Spacer(Modifier.height(16.dp))
                        Text(initProgress)
                    }
                }
            } else {
                val pv = remember {
                    PreviewView(context).apply {
                        implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                        scaleType = PreviewView.ScaleType.FIT_CENTER
                    }
                }

                // Hide so it does not flash through on back navigation
                if (scanPhase != ScanPhase.Committed) {
                    AndroidView(
                        factory = { pv },
                        modifier = Modifier
                            .fillMaxSize()
                            // Tap anywhere on the live view to scan, not
                            // just the button.
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null,
                                enabled = scanPhase == ScanPhase.Live && detectedCards.any { it.card != null }
                            ) { startReview() }
                    )
                }

                if (scanPhase == ScanPhase.Live) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val matrix = analysisToViewMatrix ?: return@Canvas
                        detectedCards.forEach { card ->
                            if (card.card != null && card.corners.size == 4) {
                                drawPath(
                                    quadPath(transformQuadCorners(card.corners, matrix)),
                                    CardOutlineColor,
                                    style = Stroke(width = 2.dp.toPx())
                                )
                            }
                        }
                    }
                } else {
                    reviewSnapshot?.let { snapshot ->
                        val elapsedMillis = (1f - reviewCountdown.value) * ScanReviewDurationMillis
                        val entrance = (elapsedMillis / ScanReviewEntranceMillis).coerceIn(0f, 1f)
                        ScanReviewOverlay(
                            cards = snapshot.cards,
                            matrix = snapshot.matrix,
                            bitmap = snapshot.bitmap,
                            entrance = entrance,
                            popScale = reviewPopScale.value,
                            onCancel = cancelReview
                        )
                    }
                }

                LaunchedEffect(Unit) {
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(pv.surfaceProvider)
                    }

                    val displayRotation = pv.display?.rotation ?: Surface.ROTATION_0

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setTargetRotation(displayRotation)
                        .build()

                    val detectorRef = detector!!

                    imageAnalysis.setAnalyzer(analyzerExecutor) { imageProxy: ImageProxy ->
                        if (scanPhase != ScanPhase.Live) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        try {
                            val bitmap = imageProxyToBitmap(imageProxy)
                            val results = detectorRef.detect(bitmap)
                            val correctionMatrix = getCorrectionMatrix(imageProxy, pv)
                            latestBitmap.record(bitmap) { reviewSnapshot?.bitmap }
                            analysisToViewMatrix = correctionMatrix
                            currentDetection = results
                            detectedCards = results?.detectedCards ?: emptyList()
                        } catch (e: Exception) {
                            Log.e("CameraScanScreen", "Detection error", e)
                        } finally {
                            imageProxy.close()
                        }
                    }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalysis
                        )
                    } catch (e: Exception) {
                        Log.e("CameraScanScreen", "Camera bind failed", e)
                        error = "$cameraStartFailedText: ${e.message}"
                    }
                }
            }

            CenterAlignedTopAppBar(
                title = { Text(stringResource(R.string.camera_title)) },
                navigationIcon = {
                    // Disabled during the review pause
                    IconButton(onClick = onBack, enabled = scanPhase == ScanPhase.Live) {
                        Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = if (detector == null) Color.Transparent else Color.Black.copy(alpha = 0.3f),
                    titleContentColor = if (detector == null) MaterialTheme.colorScheme.onSurface else Color.White,
                    navigationIconContentColor = if (detector == null) MaterialTheme.colorScheme.onSurface else Color.White
                ),
                windowInsets = WindowInsets.statusBars
            )

            if (detector != null) {
                val scanText = stringResource(R.string.camera_scan)
                val hintTemplate = stringResource(R.string.camera_hint, scanText)
                val hintText = buildAnnotatedString {
                    val idx = hintTemplate.indexOf(scanText)
                    append(hintTemplate.substring(0, idx))
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                        append(scanText)
                    }
                    append(hintTemplate.substring(idx + scanText.length))
                }
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.6f))
                            )
                        )
                        .navigationBarsPadding()
                        .padding(horizontal = 24.dp)
                        .padding(top = 40.dp, bottom = 32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    if (scanPhase == ScanPhase.Live) {
                        Text(
                            text = hintText,
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White,
                            textAlign = TextAlign.Center
                        )
                        Spacer(Modifier.height(16.dp))
                        Button(
                            onClick = startReview,
                            enabled = detectedCards.any { it.card != null }
                        ) {
                            Text(stringResource(R.string.camera_scan))
                        }
                    } else {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(32.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            IconButton(
                                onClick = cancelReview,
                                modifier = Modifier
                                    .size(48.dp)
                                    .background(Color.White.copy(alpha = 0.15f), CircleShape)
                            ) {
                                Icon(
                                    Icons.Rounded.Close,
                                    contentDescription = stringResource(R.string.picker_cancel),
                                    tint = Color.White
                                )
                            }

                            // The checkmark confirms; the X and
                            // the rest of the screen cancel instead.
                            IconButton(
                                onClick = commitReview,
                                modifier = Modifier.size(64.dp)
                            ) {
                                Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {
                                    CircularProgressIndicator(
                                        progress = { reviewCountdown.value.coerceIn(0f, 1f) },
                                        modifier = Modifier.fillMaxSize(),
                                        color = Color.White,
                                        trackColor = Color.White.copy(alpha = 0.25f),
                                        strokeWidth = 3.dp,
                                        gapSize = 0.dp
                                    )
                                    Icon(
                                        Icons.Rounded.Check,
                                        contentDescription = stringResource(R.string.camera_review_confirm),
                                        tint = Color.White
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(stringResource(R.string.camera_permission_required))
            Spacer(Modifier.height(16.dp))
            Button(onClick = { cameraPermissionState.launchPermissionRequest() }) {
                Text(stringResource(R.string.camera_grant_permission))
            }
        }
    }
}

/**
 * Creates a matrix that maps coordinates from the analysis bitmap (already
 * rotated to natural/upright orientation) to the PreviewView's FIT_CENTER
 * rendered region. Only FIT_CENTER scale and letterbox offset are needed
 * here; sensor rotation is already baked into the bitmap, and therefore into
 * every detected corner/arrangement coordinate, so there's no separate
 * rotation component to apply on top.
 */
private fun getCorrectionMatrix(imageProxy: ImageProxy, previewView: PreviewView): Matrix {
    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
    val imgW: Float
    val imgH: Float
    when (rotationDegrees) {
        90, 270 -> { imgW = imageProxy.height.toFloat(); imgH = imageProxy.width.toFloat() }
        else    -> { imgW = imageProxy.width.toFloat();  imgH = imageProxy.height.toFloat() }
    }

    val viewW = previewView.width.toFloat()
    val viewH = previewView.height.toFloat()

    // FIT_CENTER rendered region within the PreviewView
    val scale = minOf(viewW / imgW, viewH / imgH)
    val renderedW = imgW * scale
    val renderedH = imgH * scale
    val offsetX = (viewW - renderedW) / 2f
    val offsetY = (viewH - renderedH) / 2f

    val matrix = Matrix()
    matrix.postScale(scale, scale)
    matrix.postTranslate(offsetX, offsetY)
    return matrix
}

/**
 * Copies the ImageAnalysis buffer into a Bitmap rotated to natural/upright
 * orientation, so detection and every downstream consumer of its
 * corner/arrangement coordinates see the same canonically-oriented image the
 * native detector expects.
 */
private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
    val buffer = imageProxy.planes[0].buffer
    val width = imageProxy.width
    val height = imageProxy.height

    val rawBitmap = createBitmap(width, height)
    buffer.rewind()
    rawBitmap.copyPixelsFromBuffer(buffer)

    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
    if (rotationDegrees == 0) {
        return rawBitmap
    }
    val rotationMatrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    val rotatedBitmap = Bitmap.createBitmap(rawBitmap, 0, 0, width, height, rotationMatrix, true)
    rawBitmap.recycle()
    return rotatedBitmap
}

private fun copyAssetToFile(context: Context, assetName: String): File {
    val dir = File(context.cacheDir, "models")
    if (!dir.exists()) dir.mkdirs()
    val destinationFile = File(dir, assetName.substringAfterLast("/"))

    try {
        val assetManager = context.assets
        val assetSize = try {
            assetManager.openFd(assetName).use { it.length }
        } catch (e: Exception) {
            assetManager.open(assetName).use { it.available().toLong() }
        }

        if (destinationFile.exists() && destinationFile.length() == assetSize && assetSize > 0) {
            return destinationFile
        }
    } catch (e: Exception) {
        if (destinationFile.exists() && destinationFile.length() > 0) {
            return destinationFile
        }
    }

    context.assets.open(assetName).use { input ->
        FileOutputStream(destinationFile).use { output ->
            input.copyTo(output)
        }
    }
    return destinationFile
}
