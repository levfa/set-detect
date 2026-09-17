package com.fabianleven.setdetect.ui

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.graphics.createBitmap
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.DetectedCard
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.NativeSetDetector
import com.fabianleven.setdetect.domain.SetCard
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

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
    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    var analysisToViewMatrix by remember { mutableStateOf<Matrix?>(null) }
    
    val scope = rememberCoroutineScope()
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }

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
                val pv = remember { PreviewView(context).apply {
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                } }
                previewView = pv

                AndroidView(
                    factory = { pv },
                    modifier = Modifier.fillMaxSize()
                )

                Canvas(modifier = Modifier.fillMaxSize()) {
                    val matrix = analysisToViewMatrix ?: return@Canvas
                    
                    detectedCards.forEach { card ->
                        if (card.card != null && card.corners.size == 4) {
                            val path = Path().apply {
                                val pts = card.corners.map { corner ->
                                    val src = floatArrayOf(corner.x, corner.y)
                                    val dst = floatArrayOf(0f, 0f)
                                    matrix.mapPoints(dst, src)
                                    PointF(dst[0], dst[1])
                                }
                                moveTo(pts[0].x, pts[0].y)
                                lineTo(pts[1].x, pts[1].y)
                                lineTo(pts[2].x, pts[2].y)
                                lineTo(pts[3].x, pts[3].y)
                                close()
                            }
                            drawPath(path, Color.Yellow, style = Stroke(width = 2.dp.toPx()))
                        }
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
                        try {
                            val bitmap = imageProxyToBitmap(imageProxy)
                            val results = detectorRef.detect(bitmap)
                            val correctionMatrix = getCorrectionMatrix(imageProxy, pv)
                            bitmap.recycle()
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
                    IconButton(onClick = onBack) {
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
                        .navigationBarsPadding()
                        .padding(bottom = 32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = hintText,
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.White
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(
                        onClick = {
                            currentDetection?.let { onCardsDetected(it) }
                        },
                        enabled = detectedCards.any { it.card != null }
                    ) {
                        Text(stringResource(R.string.camera_scan))
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
 * Creates a matrix that maps coordinates from the analysis bitmap produced by
 * imageProxyToBitmap() (already rotated to natural/upright orientation)
 * to the PreviewView's FIT_CENTER rendered region. Only FIT_CENTER scale and
 * letterbox offset are needed here; sensor rotation is already baked into the
 * bitmap (and therefore into every detected corner/arrangement coordinate),
 * so there's no separate rotation component to apply on top.
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
 * corner/arrangement coordinates (including the arrangement screen, which has
 * no rotation correction of its own) see the same canonically-oriented image
 * the native detector expects, rather than the raw sensor buffer.
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