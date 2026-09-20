package com.fabianleven.setdetect.ui

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
                        IconButton(
                            onClick = { viewModel.tutorial.start() },
                            modifier = Modifier.onGloballyPositioned {
                                targets.helpButton = it.boundsInWindow()
                            }
                        ) {
                            Icon(Icons.AutoMirrored.Rounded.HelpOutline, contentDescription = stringResource(R.string.help))
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
