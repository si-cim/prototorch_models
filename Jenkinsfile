pipeline {
  agent none
  stages {
    stage('CPU') {
      parallel {
        stage('3.9'){
          agent {
            dockerfile {
              filename 'python39.Dockerfile'
              dir '.ci'
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
            dockerfile {
              filename 'python38.Dockerfile'
              dir '.ci'
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
