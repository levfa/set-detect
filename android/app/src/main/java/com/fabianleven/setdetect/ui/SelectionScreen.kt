package com.fabianleven.setdetect.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.HelpOutline
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.PhotoCamera
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.findSets
import com.fabianleven.setdetect.ui.components.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// Material3's own non-persistent tooltip auto-dismisses after a fixed,
// non-configurable BasicTooltipDefaults.TooltipDuration (1500ms), so a
// longer visible time is driven manually via an isPersistent state instead.
private const val TutorialHintDurationMillis = 3000L

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SelectionScreen(
    onDetectSets: (List<SetCard>) -> Unit,
    onNavigateToCamera: () -> Unit,
    onNavigateToSettings: () -> Unit,
    modifier: Modifier = Modifier,
    viewModel: SelectionViewModel = viewModel(factory = SelectionViewModel.Factory)
) {
    val boardCards = viewModel.boardCards
    val setCount = findSets(boardCards).size
    var showManualPicker by remember { mutableStateOf(false) }
    // Screen positions the tutorial can spotlight, set on the elements they track.
    val targets = remember { TutorialTargets() }

    Box(modifier = Modifier.fillMaxSize()) {
        Scaffold(
            topBar = {
                CenterAlignedTopAppBar(
                    // Navigation icon uses the same tint (onSurfaceVariant) as the action icons.
                    colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                        navigationIconContentColor = MaterialTheme.colorScheme.onSurfaceVariant
                    ),
                    title = {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.onGloballyPositioned {
                                targets.selectionInfo = it.boundsInWindow()
                            }
                        ) {
                            // The set count leads, larger and in the theme color;
                            // the card count is a small subtitle.
                            Text(
                                text = pluralStringResource(
                                    R.plurals.selection_sets_contained,
                                    setCount,
                                    setCount
                                ),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                text = pluralStringResource(
                                    R.plurals.selection_cards_selected,
                                    boardCards.size,
                                    boardCards.size
                                ),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.outline
                            )
                        }
                    },
                    navigationIcon = {
                        IconButton(onClick = onNavigateToSettings) {
                            Icon(Icons.Rounded.Settings, contentDescription = stringResource(R.string.settings))
                        }
                    },
                    actions = {
                        // isPersistent, so show() doesn't auto-dismiss after
                        // Material's default 1.5s; dismiss() is called
                        // manually below once TutorialHintDurationMillis
                        // elapses instead.
                        val tutorialTooltipState = rememberTooltipState(isPersistent = true)
                        var hintEmphasized by remember { mutableStateOf(false) }
                        LaunchedEffect(Unit) {
                            if (viewModel.consumeTutorialHint()) {
                                hintEmphasized = true
                                launch { tutorialTooltipState.show() }
                                delay(TutorialHintDurationMillis)
                                tutorialTooltipState.dismiss()
                                hintEmphasized = false
                            }
                        }
                        // Purely informational: only the icon itself starts the
                        // tutorial, the tooltip just points at it.
                        TooltipBox(
                            positionProvider = TooltipDefaults.rememberTooltipPositionProvider(TooltipAnchorPosition.Below),
                            tooltip = {
                                PlainTooltip(caretShape = TooltipDefaults.caretShape()) {
                                    Text(stringResource(R.string.tutorial_hint))
                                }
                            },
                            state = tutorialTooltipState
                        ) {
                            IconButton(onClick = { viewModel.tutorial.start() }) {
                                if (hintEmphasized) {
                                    // Same pulsing pattern the tutorial's own
                                    // spotlight border uses (TutorialOverlay.kt).
                                    val pulseTransition = rememberInfiniteTransition(label = "helpIconPulse")
                                    val pulseScale by pulseTransition.animateFloat(
                                        initialValue = 1f,
                                        targetValue = 1.25f,
                                        animationSpec = infiniteRepeatable(
                                            animation = tween(500, easing = LinearOutSlowInEasing),
                                            repeatMode = RepeatMode.Reverse
                                        ),
                                        label = "scale"
                                    )
                                    Icon(
                                        Icons.AutoMirrored.Rounded.HelpOutline,
                                        contentDescription = stringResource(R.string.help),
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.graphicsLayer {
                                            scaleX = pulseScale
                                            scaleY = pulseScale
                                        }
                                    )
                                } else {
                                    Icon(Icons.AutoMirrored.Rounded.HelpOutline, contentDescription = stringResource(R.string.help))
                                }
                            }
                        }
                    }
                )
            },
            floatingActionButton = {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    FloatingActionButton(
                        onClick = { showManualPicker = true },
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.onGloballyPositioned {
                            targets.editFab = it.boundsInWindow()
                        }
                    ) {
                        Icon(
                            Icons.Rounded.Edit,
                            contentDescription = stringResource(R.string.selection_edit_cards)
                        )
                    }
                    FloatingActionButton(
                        onClick = onNavigateToCamera,
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.onGloballyPositioned {
                            targets.fab = it.boundsInWindow()
                        }
                    ) {
                        Icon(Icons.Rounded.PhotoCamera, contentDescription = stringResource(R.string.selection_scan_cards))
                    }
                }
            },
            bottomBar = {
                BottomAppBar(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Spacer(Modifier.weight(1f))
                    FilledTonalButton(
                        onClick = { onDetectSets(boardCards) },
                        enabled = setCount > 0,
                        modifier = Modifier.onGloballyPositioned {
                            targets.revealSetsButton = it.boundsInWindow()
                        }
                    ) {
                        Text(stringResource(R.string.selection_reveal_sets))
                    }
                }
            },
            modifier = modifier
        ) { innerPadding ->
            BoardContent(
                viewModel = viewModel,
                targets = targets,
                modifier = Modifier
                    .padding(innerPadding)
                    .fillMaxSize()
            )
        }

        if (showManualPicker) {
            ManualSelectionSheet(
                boardCards = boardCards,
                onToggle = { viewModel.toggleCard(it) },
                onDone = { showManualPicker = false }
            )
        }

        TutorialLayer(tutorial = viewModel.tutorial, targets = targets)
    }
}
