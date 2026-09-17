package com.fabianleven.setdetect.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.SearchOff
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.*
import com.fabianleven.setdetect.ui.components.CardView
import com.fabianleven.setdetect.ui.theme.SetDetectTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultsListScreen(
    selectedCards: List<SetCard>,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    val sets = remember(selectedCards) { findSets(selectedCards) }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {},
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                }
            )
        },
        modifier = modifier
    ) { innerPadding ->
        if (sets.isEmpty()) {
            Column(
                modifier = Modifier.fillMaxSize().padding(innerPadding).padding(horizontal = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Rounded.SearchOff,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = MaterialTheme.colorScheme.secondary.copy(alpha = 0.5f)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = stringResource(R.string.results_no_sets_found),
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(
                    top = innerPadding.calculateTopPadding() + 16.dp,
                    bottom = innerPadding.calculateBottomPadding() + 16.dp,
                    start = 24.dp,
                    end = 24.dp
                ),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    Text(
                        text = pluralStringResource(R.plurals.results_sets_found, sets.size, sets.size),
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                }
                itemsIndexed(sets) { _, set ->
                    SetItem(set = set)
                }
            }
        }
    }
}

@Composable
fun SetItem(set: List<SetCard>, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier
                .padding(12.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally)
        ) {
            set.forEach { card ->
                // Not an actual selection UI -- isSelected here just gets the
                // crisp/full-contrast card style, not the muted "unselected" one.
                CardView(card = card, isSelected = true, modifier = Modifier.size(width = 70.dp, height = 110.dp))
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ResultsListScreenPreview() {
    SetDetectTheme {
        ResultsListScreen(
            selectedCards = listOf(
                SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID),
                SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.TWO, CardShading.SOLID),
                SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.THREE, CardShading.SOLID)
            ),
            onBack = {}
        )
    }
}
