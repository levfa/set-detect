package com.fabianleven.setdetect.ui

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.gestures.animateScrollBy
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyGridState
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.HelpOutline
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.PhotoCamera
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.findDuplicateCards
import com.fabianleven.setdetect.domain.findSets
import com.fabianleven.setdetect.domain.getAllCards
import com.fabianleven.setdetect.ui.components.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SelectionScreen(
    onDetectSets: (List<SetCard>) -> Unit,
    onNavigateToCamera: () -> Unit,
    onNavigateToSettings: () -> Unit,
    modifier: Modifier = Modifier,
    viewModel: SelectionViewModel = viewModel(factory = SelectionViewModel.Factory)
) {
    val selectedCards = viewModel.selectedCards
    val setCount = findSets(selectedCards).size
    val scope = rememberCoroutineScope()
    val gridState = rememberLazyGridState()

    val tabs = listOf(
        stringResource(R.string.selection_tab_board),
        stringResource(R.string.selection_tab_deck)
    )

    // The tutorial needs to scroll the Deck tab's grid to demonstrate
    // manual selection, but that requires LazyGridState, which only exists
    // here in Compose, so it's wired in via a callback the controller
    // invokes and awaits, rather than the controller owning the grid itself.
    LaunchedEffect(viewModel.tutorial.currentStep) {
        viewModel.tutorial.onScrollRequest = {
            // A suspend callback, awaited by the controller's own demo
            // coroutine, so the reset and scroll stroke finish in order
            // before it moves on to selecting cards. This lambda body runs
            // on viewModelScope, which lacks Compose's MonotonicFrameClock
            // needed for the animation, so launch it on `scope` (which has
            // one) and join that job instead.
            scope.launch {
                // Snap to a known starting point first (silently, this isn't
                // meant to read as a scroll itself) so the demo always looks the
                // same regardless of wherever the grid happened to be scrolled to
                // when the tutorial was opened.
                gridState.scrollToItem(0)
                delay(500)
                // One natural downward scroll stroke, covering most of a
                // screenful, not a single scrollToItem jump to a specific index.
                // A jump like that doesn't read as a scroll gesture, and
                // for a card far enough down the list it can't be "reached" by
                // anything a real scroll gesture would land on either.
                val strokeDistance = gridState.layoutInfo.viewportSize.height * 0.85f
                gridState.animateScrollBy(strokeDistance, animationSpec = tween(700, easing = FastOutSlowInEasing))
                delay(500)
            }.join()

            // Which cards end up centered depends on screen size, grid
            // column count, and item height (all device-specific), so don't
            // guess an index: read gridState's real post-scroll layout and find
            // whichever row is closest to the viewport's vertical
            // center, then map that row's item indices back to cards. This
            // adapts to any device instead of being tuned for one.
            val viewportCenter = gridState.layoutInfo.viewportSize.height / 2f
            val visibleItems = gridState.layoutInfo.visibleItemsInfo
            val centeredRowOffset = visibleItems.minByOrNull { item ->
                kotlin.math.abs((item.offset.y + item.size.height / 2f) - viewportCenter)
            }?.offset?.y
            val allCards = getAllCards()
            visibleItems.filter { it.offset.y == centeredRowOffset }
                .map { allCards[it.index] }
        }
    }

    // Screen positions the tutorial can spotlight or switch tabs toward;
    // see TutorialTargets. Each field is set inline below, right at the real
    // UI element it tracks.
    val targets = remember { TutorialTargets() }

    Box(modifier = Modifier.fillMaxSize()) {
        Scaffold(
            topBar = {
                CenterAlignedTopAppBar(
                    // Match the navigation-icon tint to the action-icon tint
                    // (onSurfaceVariant) that HelpOutline already gets; the
                    // default (onSurface) is darker and looks mismatched next to it.
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
                            // Sets found is the number that matters; lead with
                            // it, larger and in the theme color, and demote the card
                            // count to a small subtitle underneath.
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
                                    selectedCards.size,
                                    selectedCards.size
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
                FloatingActionButton(
                    onClick = onNavigateToCamera,
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.onGloballyPositioned {
                        targets.fab = it.boundsInWindow()
                    }
                ) {
                    Icon(Icons.Rounded.PhotoCamera, contentDescription = null)
                }
            },
            bottomBar = {
                BottomAppBar(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Spacer(Modifier.weight(1f))
                    FilledTonalButton(
                        onClick = { viewModel.clearSelection() },
                        // Clear resets both the selection and the scan, so it
                        // should stay enabled if either one has something to
                        // clear, not just the selection.
                        enabled = selectedCards.isNotEmpty() || viewModel.scannedDetection != null
                    ) {
                        Text(stringResource(R.string.selection_clear))
                    }
                    Spacer(Modifier.width(8.dp))
                    FilledTonalButton(
                        onClick = { onDetectSets(selectedCards.toList()) },
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
            Column(modifier = Modifier.padding(innerPadding)) {
                PrimaryTabRow(
                    selectedTabIndex = viewModel.selectedTabIndex,
                    modifier = Modifier.onGloballyPositioned {
                        targets.tabRow = it.boundsInWindow()
                    }
                ) {
                    tabs.forEachIndexed { index, title ->
                        Tab(
                            selected = viewModel.selectedTabIndex == index,
                            onClick = { viewModel.selectedTabIndex = index },
                            text = { Text(title) },
                            modifier = Modifier.onGloballyPositioned {
                                if (index == BOARD_TAB_INDEX) {
                                    targets.boardTab = it.boundsInWindow()
                                } else {
                                    targets.deckTab = it.boundsInWindow()
                                }
                            }
                        )
                    }
                }

                when (viewModel.selectedTabIndex) {
                    BOARD_TAB_INDEX -> BoardTab(
                        viewModel = viewModel,
                        targets = targets,
                        modifier = Modifier.fillMaxSize()
                    )
                    DECK_TAB_INDEX -> DeckTab(
                        viewModel = viewModel,
                        state = gridState,
                        modifier = Modifier
                            .fillMaxSize()
                            .onGloballyPositioned {
                                targets.deck = it.boundsInWindow()
                            }
                    )
                }
            }
        }

        TutorialLayer(tutorial = viewModel.tutorial, targets = targets, tabLabels = tabs)
    }
}

@Composable
private fun DeckTab(
    viewModel: SelectionViewModel,
    state: LazyGridState,
    modifier: Modifier = Modifier
) {
    val allCards = remember { getAllCards() }
    val selectedCards = viewModel.selectedCards

    LazyVerticalGrid(
        columns = GridCells.Adaptive(minSize = 100.dp),
        state = state,
        contentPadding = PaddingValues(
            top = 8.dp,
            bottom = 80.dp,
            start = 8.dp,
            end = 8.dp
        ),
        modifier = modifier
    ) {
        items(allCards) { card ->
            val isSelected = selectedCards.contains(card)
            CardView(
                card = card,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(4.dp),
                isSelected = isSelected,
                onClick = { viewModel.toggleSelection(card) }
            )
        }
    }
}

@Composable
private fun BoardTab(
    viewModel: SelectionViewModel,
    targets: TutorialTargets,
    modifier: Modifier = Modifier
) {
    val detection = viewModel.scannedDetection
    val selectedCards = viewModel.selectedCards.toSet()
    // detection.arrangement.cardPoses[i] corresponds to the i-th *matched* card
    // (native's detect_cards() fits it over matched_cards, which drops any
    // detectedCards entry that failed classification (card == null) while
    // preserving order), not the i-th entry of detectedCards. Reconstruct that
    // same matched-only sequence here rather than passing the raw list
    // (with null gaps for unmatched cards) to ArrangementView, which indexes
    // cards by pose position: an unmatched card would throw every later
    // card's rendering out of alignment with its pose, hiding some and
    // misplacing others.
    val matchedCards = remember(detection) { detection?.detectedCards?.mapNotNull { it.card } ?: emptyList() }
    // Selected cards that aren't part of the current scan (or there is no scan at
    // all yet): these came from the Deck tab and would otherwise be invisible
    // here, making it look like they'd been lost.
    val manualSelectionCards = remember(selectedCards, matchedCards) {
        selectedCards.filter { it !in matchedCards }
    }
    // A card matching another one currently on the board violates the game's one-of-each
    // rule: either the same physical card is there twice, or classification
    // misread one of them. Flag it rather than silently miscounting/misreading the board.
    val duplicateCards = remember(matchedCards) { findDuplicateCards(matchedCards) }

    Column(modifier = modifier) {
        // Always visible: even with nothing selected yet, the Add tile is
        // itself the invitation to go pick some cards manually.
        Text(
            text = stringResource(R.string.selection_manual_selection),
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(start = 16.dp, top = 8.dp, bottom = 4.dp)
        )
        LazyVerticalGrid(
            columns = GridCells.Adaptive(minSize = 60.dp),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            modifier = Modifier
                .height(120.dp)
                .onGloballyPositioned { targets.manualSelection = it.boundsInWindow() }
        ) {
            item {
                AddCardsTile(
                    onClick = { viewModel.selectedTabIndex = DECK_TAB_INDEX },
                    modifier = Modifier.padding(2.dp)
                )
            }
            items(manualSelectionCards) { card ->
                CardView(
                    card = card,
                    isSelected = true,
                    onClick = { viewModel.toggleSelection(card) },
                    modifier = Modifier.padding(2.dp)
                )
            }
        }
        HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))

        Text(
            text = stringResource(R.string.selection_scan),
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(start = 16.dp, top = 8.dp, bottom = 4.dp)
        )
        Box(
            modifier = Modifier
                .weight(1f)
                .onGloballyPositioned { targets.arrangement = it.boundsInWindow() }
        ) {
            if (detection == null) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.PhotoCamera,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.outline,
                            modifier = Modifier.size(48.dp)
                        )
                        Text(
                            text = stringResource(R.string.selection_scan_empty),
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                }
            } else {
                ArrangementView(
                    arrangement = detection.arrangement,
                    cards = matchedCards,
                    selectedCards = selectedCards,
                    onToggleCard = { viewModel.toggleSelection(it) },
                    highlightedCard = viewModel.tutorial.demoCard,
                    onHighlightedCardPositioned = { targets.demoCardRect = it },
                    duplicateCards = duplicateCards,
                    modifier = Modifier.fillMaxSize().padding(16.dp)
                )
            }
            if (duplicateCards.isNotEmpty()) {
                Surface(
                    color = MaterialTheme.colorScheme.errorContainer,
                    shape = RoundedCornerShape(50),
                    shadowElevation = 4.dp,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(16.dp)
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
        }
    }
}

// A card-shaped tile at the start of the "manual selection" row that jumps to
// the Deck tab, so adding more cards manually is one tap away from here.
@Composable
private fun AddCardsTile(
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        onClick = onClick,
        modifier = modifier.aspectRatio(5f / 7f),
        shape = RoundedCornerShape(12),
        colors = CardDefaults.cardColors(containerColor = Color.Transparent),
        border = BorderStroke(2.dp, MaterialTheme.colorScheme.outline)
    ) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Icon(
                imageVector = Icons.Rounded.Add,
                contentDescription = stringResource(R.string.selection_add_cards),
                tint = MaterialTheme.colorScheme.outline
            )
        }
    }
}
