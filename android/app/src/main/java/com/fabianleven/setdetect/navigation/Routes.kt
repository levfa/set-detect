package com.fabianleven.setdetect.navigation

import androidx.navigation3.runtime.NavKey
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.SetCard
import kotlinx.serialization.Serializable

@Serializable
sealed interface Route : NavKey {
    @Serializable
    data object Selection : Route

    @Serializable
    data object Settings : Route

    @Serializable
    data object About : Route

    @Serializable
    data object Licenses : Route

    @Serializable
    data object PrivacyPolicy : Route

    @Serializable
    data object CameraScan : Route

    @Serializable
    data class Results(val selectedCards: List<SetCard>) : Route
}
