pipeline {
    agent any // Tells Jenkins to run this on any available "worker"

    options {
        timestamps() // Adds clock times to the logs
    }

    stages {
        stage("Checkout") {
            steps {
                // Jenkins automatically clones the GitHub repo here
                checkout([$class: 'GitSCM', 
                branches: [[name: '*/RecidualExtraction']], 
                userRemoteConfigs: [[url: 'https://github.com/lukasfdahl/P4.git', 
                credentialsId: 'github-kls-bot']]])
            }
        }

        stage("Build Docker Image") {
            steps {
                echo "Building the Docker container:"
                sh "docker build -f RecidualExtraction/Dockerfile -t video-extractor-container:${env.BUILD_ID} ./RecidualExtraction"
            }
        }
        stage("Run Recidual/MV/Type extraction") {
            steps {
                echo "Running basic test to see if library is working:"
                sh "docker run --rm -v \'${WORKSPACE}\':/app/test video-extractor-container:${env.BUILD_ID} python3 /app/test/extractor.py"
                archiveArtifacts artifacts: 'output_data/*.*', allowEmptyArchive: false // To save the output files for download by us
            }
        }
    }

    post {
        always {
            echo "Cleaning up old Docker images:"
            sh "docker rmi video-extractor-container:${env.BUILD_ID} || true"
        }
        success {
            echo "Pipeline passed"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}
