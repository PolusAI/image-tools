pipeline {
    agent {
        node { label 'linux && build && aws' }
    }
    environment {
        PROJECT_URL = 'https://github.com/polusai/polus-plugins'
    }
    triggers {
        pollSCM('H/5 * * * *')
    }
    stages {
        stage('Build Version') {
            steps{
                script {
                    BUILD_VERSION_GENERATED = VersionNumber(
                        versionNumberString: 'v${BUILD_YEAR, XX}.${BUILD_MONTH, XX}${BUILD_DAY, XX}.${BUILDS_TODAY}',
                        projectStartDate: '1970-01-01',
                        skipFailedBuilds: false)
                    currentBuild.displayName = BUILD_VERSION_GENERATED
                    env.BUILD_VERSION = BUILD_VERSION_GENERATED
                }
            }
        }
        stage('Checkout source code') {
            steps {
                cleanWs()
                checkout scm
            }
        }
        stage('Build Docker images') {
            steps {
                script {
                    configFileProvider([configFile(fileId: 'update-docker-description', targetLocation: 'update.sh')]) {
                        // List all directories, each directory contains a plugin
                        def pluginDirectories = """${sh (
                            script: "ls -d */",
                            returnStdout: true
                        )}"""
                        // Iterate over each plugin directory
                        pluginDirectories.split().each { repo ->
                            // Truncate hanging "/" for each directory
                            def pluginName = repo.getAt(0..(repo.length() - 2))
                            // Check if VERSION file for each plugin file has changed
                            def isChanged = "0"

                            if (env.GIT_PREVIOUS_SUCCESSFUL_COMMIT) {
                                isChanged = """${sh (
                                    script: "git diff --name-only ${GIT_PREVIOUS_SUCCESSFUL_COMMIT} ${GIT_COMMIT} | grep ${pluginName}/VERSION",
                                    returnStatus: true
                                )}"""
                            }
                            if (isChanged == "0" && pluginName != "utils") {
                                dir("${WORKSPACE}/${pluginName}") {
                                    def dockerVersion = readFile(file: 'VERSION').trim()
                                    docker.withRegistry('https://registry-1.docker.io/v2/', 'f16c74f9-0a60-4882-b6fd-bec3b0136b84') {
                                        def image = docker.build("labshare/${pluginName}", '--no-cache ./')
                                        image.push()
                                        image.push(dockerVersion)
                                    }
                                    
                                    env.PROJECT_NAME = "${pluginName}"
                                    env.FULL_DESC = readFile(file: 'README.md')
                                    env.BRIEF_DESC = "${PROJECT_URL}/tree/master/${PROJECT_NAME}"
                                }
                                withCredentials([usernamePassword(credentialsId: 'f16c74f9-0a60-4882-b6fd-bec3b0136b84', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PW')]) {
                                    sh "sh ./update.sh"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}