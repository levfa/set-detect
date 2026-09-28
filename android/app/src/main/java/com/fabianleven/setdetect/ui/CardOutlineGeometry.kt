package com.fabianleven.setdetect.ui

import android.graphics.Matrix
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path

// Used for both the live detection outline and the review pause's outline.
val CardOutlineColor = Color.Yellow

/**
 * Maps a card's corners (in analysis-bitmap space) to on-screen points via
 * the given analysis-to-view matrix.
 */
fun transformQuadCorners(
    corners: List<com.fabianleven.setdetect.domain.PointF>,
    matrix: Matrix
): List<Offset> {
    return corners.map { corner ->
        val src = floatArrayOf(corner.x, corner.y)
        val dst = floatArrayOf(0f, 0f)
        matrix.mapPoints(dst, src)
        Offset(dst[0], dst[1])
    }
}

/** Builds a closed path through [points] in order. */
fun quadPath(points: List<Offset>): Path {
    return Path().apply {
        moveTo(points[0].x, points[0].y)
        for (i in 1 until points.size) lineTo(points[i].x, points[i].y)
        close()
    }
}

/**
 * Pushes each point directly away from the quad's centroid by [amountPx], so
 * a shape built from the result sits just outside the original quad rather
 * than tracing its exact edge.
 */
fun expandQuad(points: List<Offset>, amountPx: Float): List<Offset> {
    val centroid = quadCentroid(points)
    return points.map { point ->
        val direction = point - centroid
        val length = direction.getDistance()
        if (length < 0.001f) point else point + direction * (amountPx / length)
    }
}

/** Scales each point toward/away from the quad's own centroid by [scale]. */
fun scaleQuad(points: List<Offset>, scale: Float): List<Offset> {
    val centroid = quadCentroid(points)
    return points.map { centroid + (it - centroid) * scale }
}

private fun quadCentroid(points: List<Offset>): Offset = Offset(
    points.sumOf { it.x.toDouble() }.toFloat() / points.size,
    points.sumOf { it.y.toDouble() }.toFloat() / points.size
)
