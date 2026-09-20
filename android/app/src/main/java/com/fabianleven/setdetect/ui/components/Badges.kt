package com.fabianleven.setdetect.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R

// Small round icon meant to sit in a card's corner; the caller positions it.
@Composable
fun CornerBadge(
    imageVector: ImageVector,
    contentDescription: String,
    color: Color,
    onColor: Color,
    iconSize: Dp,
    modifier: Modifier = Modifier
) {
    Icon(
        imageVector = imageVector,
        contentDescription = contentDescription,
        tint = onColor,
        modifier = modifier
            .background(color, CircleShape)
            .padding(iconSize / 6)
            .size(iconSize)
    )
}

@Composable
fun DuplicateBadge(iconSize: Dp, modifier: Modifier = Modifier) {
    CornerBadge(
        imageVector = Icons.Rounded.Warning,
        contentDescription = stringResource(R.string.selection_duplicate_card),
        color = MaterialTheme.colorScheme.error,
        onColor = MaterialTheme.colorScheme.onError,
        iconSize = iconSize,
        modifier = modifier
    )
}

@Composable
fun DuplicateBanner(modifier: Modifier = Modifier) {
    Surface(
        color = MaterialTheme.colorScheme.errorContainer,
        shape = RoundedCornerShape(50),
        shadowElevation = 4.dp,
        modifier = modifier
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
        ) {
            Icon(
                imageVector = Icons.Rounded.Warning,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onErrorContainer,
                modifier = Modifier.size(16.dp)
            )
            Spacer(Modifier.width(6.dp))
            Text(
                text = stringResource(R.string.selection_duplicate_warning),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onErrorContainer
            )
        }
    }
}
