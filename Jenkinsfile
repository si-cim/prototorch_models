pipeline {
  agent none
  stages {
    stage('Unit Tests') {
          agent {
            dockerfile {
              filename 'python310.Dockerfile'
              dir '.ci'
            }

          }
          environment {
            PATH = "/home/jenkins/.local/bin:${env.PATH}"
          }
          steps {
            sh 'pip install pip --upgrade --progress-bar off'
            sh 'pip install .[all] --progress-bar off'
            sh 'pytest -v --junitxml=reports/result.xml'
          }
        post {
          always {
              junit 'reports/**/*.xml'
          }
        }
    }

    stage('CPU Examples') {
      parallel {
        stage('3.10') {
          agent {
            dockerfile {
              filename 'python310.Dockerfile'
              dir '.ci'
            }

          }
          steps {
            sh 'pip install pip --upgrade --progress-bar off'
            sh 'pip install .[all] --progress-bar off'
            sh './tests/test_examples.sh examples'
          }
        }

        stage('3.9') {
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

        stage('3.8') {
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

        stage('3.7') {
          agent {
            dockerfile {
              filename 'python37.Dockerfile'
              dir '.ci'
            }

          }
          steps {
            sh 'pip install pip --upgrade --progress-bar off'
            sh 'pip install .[all] --progress-bar off'
            sh './tests/test_examples.sh examples'
          }
        }

        stage('3.6') {
          agent {
            dockerfile {
              filename 'python36.Dockerfile'
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

    stage('GPU Examples') {
      agent {
        dockerfile {
          filename 'gpu.Dockerfile'
          dir '.ci'
          args '--gpus 1'
        }

      }
      steps {
        sh 'pip install -U pip --progress-bar off'
        sh 'pip install .[all] --progress-bar off'
        sh './tests/test_examples.sh examples --gpu'
      }
    }

  }
}
