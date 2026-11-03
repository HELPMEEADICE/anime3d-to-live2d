plugins {
    kotlin("jvm") version "2.4.10"
    kotlin("plugin.serialization") version "2.4.10"
    application
}
repositories { mavenCentral() }
kotlin {
    jvmToolchain(26)
    compilerOptions { freeCompilerArgs.add("-Xno-optimize") }
    sourceSets.main { kotlin.srcDir("../vendor/psd2live/umamo/src/main/kotlin") }
}
dependencies {
    implementation(kotlin("reflect"))
    implementation("org.jdom:jdom:1.1.3")
    implementation("com.squareup.okio:okio:3.17.0")
    implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.8.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation(platform("org.lwjgl:lwjgl-bom:3.4.2"))
    implementation("org.lwjgl:lwjgl")
    implementation("org.lwjgl:lwjgl-opengl")
}
application { mainClass.set("popy.ExportKt"); applicationDefaultJvmArgs = listOf("-Xmx24g", "-Djava.awt.headless=true") }
