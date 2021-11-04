pipeline {
  agent none
  stages {
    stage('CPU') {
      parallel {
        stage('3.9'){
          agent {
            docker {
              image 'python:3.9'
              args '--user 0:0'
            }
          }
          steps {
            sh 'pip install pip --upgrade --progress-bar off'
            sh 'pip install .[all] --progress-bar off'
            sh './tests/test_examples.sh examples'
          }
        }
        stage('3.8'){
          agent {
            docker {
              image 'python:3.8'
              args '--user 0:0'
            }
          }
          steps {
            sh 'pip install pip --upgrade --progress-bar off'
            sh 'pip install .[all] --progress-bar off'
            sh './tests/test_examples.sh examples'
          }
        }
      }
    }
  }
}
