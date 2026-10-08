package com.fabianleven.setdetect.ui

import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.os.LocaleListCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.fabianleven.setdetect.R

private data class LanguageOption(val tag: String?, val labelRes: Int)

// tag == null means "follow the system language", represented to
// AppCompatDelegate as an empty locale list, not a specific one.
private val languageOptions = listOf(
    LanguageOption(null, R.string.settings_language_system_default),
    LanguageOption("en", R.string.settings_language_english),
    LanguageOption("de", R.string.settings_language_german),
    LanguageOption("es", R.string.settings_language_spanish),
    LanguageOption("fr", R.string.settings_language_french)
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    onNavigateToAbout: () -> Unit,
    modifier: Modifier = Modifier,
    viewModel: SettingsViewModel = viewModel(factory = SettingsViewModel.Factory)
) {
    val showTutorialHint by viewModel.showTutorialHint.collectAsState()

    // AppCompatDelegate is the source of truth for the chosen locale (it
    // persists itself); selecting a new one triggers an activity recreation,
    // which re-reads it below. Also re-read on every resume, in case the
    // locale changed without a recreate (e.g. via the system's per-app
    // language screen while this screen stayed alive in the background).
    fun readSelectedLanguageTag() =
        AppCompatDelegate.getApplicationLocales().let { if (it.isEmpty) null else it[0]?.language }

    var selectedLanguageTag by remember { mutableStateOf(readSelectedLanguageTag()) }
    var showLanguageDialog by remember { mutableStateOf(false) }

    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                selectedLanguageTag = readSelectedLanguageTag()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    val selectedLanguageLabelRes = languageOptions.first { it.tag == selectedLanguageTag }.labelRes

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                windowInsets = WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal),
                title = { Text(stringResource(R.string.settings)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                }
            )
        },
        modifier = modifier
    ) { innerPadding ->
        Column(modifier = Modifier.fillMaxSize().padding(innerPadding)) {
            ListItem(
                headlineContent = { Text(stringResource(R.string.settings_show_tutorial_hint)) },
                trailingContent = {
                    Switch(
                        checked = showTutorialHint,
                        onCheckedChange = { viewModel.setShowTutorialHint(it) }
                    )
                },
                modifier = Modifier.clickable { viewModel.setShowTutorialHint(!showTutorialHint) }
            )

            HorizontalDivider()

            // A single row opening a dialog, rather than an inline list of
            // radio buttons, which doesn't scale as more languages are
            // added, whereas the dialog's LazyColumn does.
            ListItem(
                headlineContent = { Text(stringResource(R.string.settings_language)) },
                supportingContent = { Text(stringResource(selectedLanguageLabelRes)) },
                trailingContent = { Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null) },
                modifier = Modifier.clickable { showLanguageDialog = true }
            )

            HorizontalDivider()

            ListItem(
                headlineContent = { Text(stringResource(R.string.about)) },
                leadingContent = { Icon(Icons.Rounded.Info, contentDescription = null) },
                trailingContent = { Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null) },
                modifier = Modifier.clickable(onClick = onNavigateToAbout)
            )
        }
    }

    if (showLanguageDialog) {
        LanguagePickerDialog(
            selectedTag = selectedLanguageTag,
            onSelect = { tag ->
                selectedLanguageTag = tag
                val locales = if (tag == null) {
                    LocaleListCompat.getEmptyLocaleList()
                } else {
                    LocaleListCompat.forLanguageTags(tag)
                }
                AppCompatDelegate.setApplicationLocales(locales)
                showLanguageDialog = false
            },
            onDismiss = { showLanguageDialog = false }
        )
    }
}

@Composable
private fun LanguagePickerDialog(
    selectedTag: String?,
    onSelect: (String?) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.settings_language)) },
        text = {
            // LazyColumn, so this keeps working however many languages get
            // added later.
            LazyColumn(modifier = Modifier.selectableGroup()) {
                items(languageOptions) { option ->
                    val selected = option.tag == selectedTag
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(selected = selected, onClick = { onSelect(option.tag) })
                            .padding(vertical = 12.dp)
                    ) {
                        RadioButton(selected = selected, onClick = null)
                        Spacer(Modifier.width(12.dp))
                        Text(stringResource(option.labelRes))
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.close))
            }
        }
    )
}
