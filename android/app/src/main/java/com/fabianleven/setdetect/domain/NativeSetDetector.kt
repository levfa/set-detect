package com.fabianleven.setdetect.domain

import android.graphics.Bitmap
import kotlinx.serialization.Serializable

@Serializable
data class PointF(val x: Float, val y: Float)

@Serializable
data class DetectedCard(
    val card: SetCard?,
    val corners: List<PointF>,
    val confidence: Float
)

@Serializable
data class RotatedRect(
    val centerX: Float,
    val centerY: Float,
    val angle: Float
)

@Serializable
data class CardsArrangement(
    val cardPoses: List<RotatedRect>,
    val cardZOrders: List<Int>,
    val cardWidth: Float,
    val cardHeight: Float
)

@Serializable
data class Detection(
    val detectedCards: List<DetectedCard>,
    val arrangement: CardsArrangement
)

class NativeSetDetector(
    cornerModelPath: String,
    classModelPath: String
) {
    private var nativePtr: Long = 0

    init {
        nativePtr = initNative(cornerModelPath, classModelPath)
        if (nativePtr == 0L) {
            throw IllegalStateException("Failed to initialize native detector. Check logs for details.")
        }
    }

    @Synchronized
    fun detect(bitmap: Bitmap): Detection? {
        if (nativePtr == 0L) return null
        return detectNative(nativePtr, bitmap)
    }

    fun close() {
        val ptr: Long
        synchronized(this) {
            ptr = nativePtr
            // Detach immediately so subsequent detect() calls become no-ops
            // without waiting for native teardown (which joins the inference
            // worker and may take a full CPU inference on slow devices).
            nativePtr = 0L
        }
        if (ptr != 0L) {
            Thread {
                synchronized(this) {
                    closeNative(ptr)
                }
            }.start()
        }
    }

    private external fun initNative(cornerModelPath: String, classModelPath: String): Long
    private external fun detectNative(nativePtr: Long, bitmap: Bitmap): Detection?
    private external fun closeNative(nativePtr: Long)

    companion object {
        init {
            try {
                System.loadLibrary("onnxruntime")
                System.loadLibrary("setdetect_jni")
            } catch (e: UnsatisfiedLinkError) {
                // Log or handle error
            }
        }
    }
}