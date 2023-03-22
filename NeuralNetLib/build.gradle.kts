plugins {
    id("java-library")
    id("org.jetbrains.kotlin.jvm")
    `maven-publish`
}

group = "com.github.pustoslov"
version = "1.0.3"

java {
    sourceCompatibility = JavaVersion.VERSION_1_7
    targetCompatibility = JavaVersion.VERSION_1_7
}

repositories{
    mavenCentral()
    maven("https://jitpack.io")
}

dependencies{
    implementation("org.jetbrains.kotlinx:multik-core:0.2.1")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.1")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "com.github.pustoslov"
            artifactId = "neural-network"
            version = "1.0"

            from(components["java"])
        }
    }
}
