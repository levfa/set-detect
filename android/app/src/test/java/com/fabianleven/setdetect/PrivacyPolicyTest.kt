package com.fabianleven.setdetect

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

// The Play Store listing links to
// https://github.com/levfa/set-detect/blob/main/android-publishing/privacy-policy.md,
// so that path must keep existing in the repository.
class PrivacyPolicyTest {
    private fun repoRoot(): File {
        var dir: File? = File(System.getProperty("user.dir")).absoluteFile
        while (dir != null && !File(dir, ".git").exists()) dir = dir.parentFile
        return checkNotNull(dir) { "repository root not found" }
    }

    @Test
    fun privacyPolicyExistsAtPublishedPath() {
        val policy = File(repoRoot(), "android-publishing/privacy-policy.md")
        assertTrue("${policy.path} must exist: the Play Store links to it", policy.isFile)
        assertTrue("${policy.path} must not be empty", policy.readText().isNotBlank())
    }
}
