package com.fabianleven.setdetect.navigation

import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.navigation3.rememberViewModelStoreNavEntryDecorator
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.runtime.rememberSaveableStateHolderNavEntryDecorator
import androidx.navigation3.ui.NavDisplay
import com.fabianleven.setdetect.ui.*

@Composable
fun SetDetectNavGraph(modifier: Modifier = Modifier) {
    val backStack = rememberNavBackStack(Route.Selection)
    val selectionViewModel: SelectionViewModel = viewModel(
        factory = SelectionViewModel.Factory
    )

    NavDisplay(
        backStack = backStack,
        onBack = { if (backStack.size > 1) backStack.removeAt(backStack.size - 1) },
        modifier = modifier,
        entryDecorators = listOf(
            rememberSaveableStateHolderNavEntryDecorator(),
            rememberViewModelStoreNavEntryDecorator()
        ),
        entryProvider = entryProvider {
            entry<Route.Selection> {
                SelectionScreen(
                    viewModel = selectionViewModel,
                    onDetectSets = { selectedCards ->
                        backStack.add(Route.Results(selectedCards))
                    },
                    onNavigateToCamera = {
                        backStack.add(Route.CameraScan)
                    },
                    onNavigateToSettings = {
                        backStack.add(Route.Settings)
                    }
                )
            }
            entry<Route.Settings> {
                SettingsScreen(
                    onBack = { backStack.removeAt(backStack.size - 1) },
                    onNavigateToAbout = {
                        backStack.add(Route.About)
                    }
                )
            }
            entry<Route.About> {
                AboutScreen(
                    onBack = { backStack.removeAt(backStack.size - 1) },
                    onNavigateToLicenses = {
                        backStack.add(Route.Licenses)
                    },
                    onNavigateToPrivacy = {
                        backStack.add(Route.PrivacyPolicy)
                    }
                )
            }
            entry<Route.Licenses> {
                LicensesScreen(
                    onBack = { backStack.removeAt(backStack.size - 1) }
                )
            }
            entry<Route.PrivacyPolicy> {
                PrivacyPolicyScreen(
                    onBack = { backStack.removeAt(backStack.size - 1) }
                )
            }
            entry<Route.CameraScan> {
                CameraScanScreen(
                    onCardsDetected = { detection ->
                        selectionViewModel.updateScannedDetection(detection)
                        backStack.removeAt(backStack.size - 1)
                    },
                    onBack = { backStack.removeAt(backStack.size - 1) }
                )
            }
            entry<Route.Results> { key ->
                ResultsListScreen(
                    selectedCards = key.selectedCards,
                    onBack = { backStack.removeAt(backStack.size - 1) }
                )
            }
        }
    )
}
