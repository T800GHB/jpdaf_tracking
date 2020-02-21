#include "tracker.h"
using namespace JPDAFTracker;

void Tracker::drawTracks(cv::Mat &_img) const {
  for(const auto& track : tracks_) {
    if(track->getId() != -1) {
      const cv::Point& p = track->getLastPrediction();
      cv::circle(_img, p, 8, track->getColor(), -1);
//      cv::ellipse(_img, p, cv::Size(25, 50), 0, 0, 360, track->getColor(), 3);
      cv::putText(_img, std::to_string(track->getId()), p, cv::FONT_HERSHEY_SIMPLEX,
		0.50, cv::Scalar(0, 255, 0), 2, CV_AA);
    }
  }
}

Eigen::MatrixXf Tracker::joint_probability(const Matrices& _association_matrices,
						const Vec2f& selected_detections) {
  uint32_t hyp_num = _association_matrices.size();
  Eigen::VectorXf Pr(hyp_num); // Probability list for every hypothesis
  uint32_t num_dets = selected_detections.size();  // Number of selected detection
  uint32_t track_size = tracks_.size();
  float prior;
  
  //Compute the total volume
  float V = 0.;
  for(const auto& track : tracks_) {
    V += track->getEllipseVolume();
  }

  for(uint32_t i = 0; i < hyp_num; ++i) {
    //I assume that all the measurments can be false alarms
    int32_t false_alarms = num_dets ;
    float N = 1.;
    //For each measurement j: I compute the measurement indicator ( tau(j, X) ) 
    // and the target detection indicator ( lambda(t, X) ) 
    for(uint32_t j = 0; j < num_dets; ++j) {
      //Compute the MEASURAMENT ASSOCIATION INDICATOR      
      const Eigen::MatrixXf& A_matrix = _association_matrices.at(i).block(j, 1, 1, track_size);
      const int32_t& mea_indicator = A_matrix.sum();     

      if(mea_indicator == 1) {
        //Update the total number of wrong measurements in X
        --false_alarms;
        //Detect which track is associated to the measurement j
        //and compute the probability
        for(uint32_t notZero = 0; notZero < track_size; ++notZero) {
          if(A_matrix(0, notZero) == 1) {
            const Eigen::Vector2f& z_predict = tracks_.at(notZero)->getLastPredictionEigen();
            const Eigen::Matrix2f& S = tracks_.at(notZero)->S();
            const Eigen::Vector2f& diff = selected_detections.at(j) - z_predict;
            // Eigen calculate normal Mahalanobis distance
            const float& eb = diff.transpose() * S.inverse() * diff;
            // Opencv calculate sqrt of normal Mahalanobis distance
            const float& cb = OpencvMahalanobis(z_predict, selected_detections.at(j), S);

            // Detection obey Gaussian distribution. In the standard Gaussian distribution, there is 0.5 before b
            // But, I think author want to amplify the covariance, cover more measurements
            // Assocation gate use sqrt of Mahalanobis to filter detection

            // sqrt((2*CV_PI*S).determinant()) is equal to pow(pow((2 * CV_PI), 0.5), 2) * pow(S.determinant(), 0.5)

            // The original author calculation, different from standard formula, i do not know why
//             N = N / sqrt((2*CV_PI*S).determinant())*exp(-cb);

            // Calcuation should be this, according to standard Gaussian distribution formula, have samilar result
            N = N / sqrt((2*CV_PI*S).determinant())*exp(-0.5 * eb);
          }
        }
      }
    }
    // False alarm obey uniform distribution
    const float& likelyhood = N / float(std::pow(V, false_alarms));
       
    if(param_.pd == 1) {
      prior = 1.;
    } else {
      //Compute the TARGET ASSOCIATION INDICATOR
      prior = 1.;
      for(uint32_t j = 0; j < track_size; ++j) {
        const Eigen::MatrixXf& target_matrix = _association_matrices.at(i).col(j+1);
        const int32_t& target_indicator = target_matrix.sum();
        prior = prior * std::pow(param_.pd, target_indicator) * std::pow((1 - param_.pd), (1 - target_indicator));
      }
    }
    
    // Compute the number of false alarm events in current hypothesis for which the same target set has been detected
    // Factorial of reversed false alarm

    int32_t a = 1;
    for(int32_t j = 1; j <= false_alarms; ++j) {
      a = a * j;
    }
    //
    Pr(i) = a * likelyhood * prior;
  }
  
  const float& prSum = Pr.sum();
  
  if(prSum != 0.) {
      Pr = Pr / prSum; //normalization
  }
    
  //Compute Beta Coefficients
  Eigen::MatrixXf beta(num_dets + 1, track_size);
  beta = Eigen::MatrixXf::Zero(num_dets + 1, track_size);
   
  Eigen::VectorXf sumBeta(track_size);
  sumBeta.setZero();

  for(uint32_t i = 0; i < track_size; ++i) {
    for(uint32_t j = 0; j < num_dets; ++j) {
      for(uint32_t k = 0; k < hyp_num; ++k) {
        // Probability of j-th measurement associate to i-th tracker, accumulate by all hypotheses
        beta(j, i) += Pr(k) * _association_matrices.at(k)(j, i+1);
      }
      sumBeta(i) += beta(j, i);
    }
    sumBeta(i) = 1 - sumBeta(i);    // Probability of no measurement associate to current tracker
  }
  
  beta.row(num_dets) = sumBeta;

  return beta;
}


Tracker::Matrices Tracker::generate_hypothesis(const cv::Mat &_q) {
  uint32_t num_det = _q.rows;
  //All the measurements can be generated by the clutter track
  Eigen::MatrixXf A_Matrix(_q.rows, _q.cols); 
  A_Matrix = Eigen::MatrixXf::Zero(_q.rows, _q.cols);
  // First column set all one means all false alarm
  A_Matrix.col(0).setOnes();
  // A_Matrix could be a sparse matrix, less memory, than delete limitation of hypothesis by MAX_ASSOC
  Matrices tmp_association_matrices(MAX_ASSOC, A_Matrix);
  
  uint32_t hyp_num = 0;
  //Generating all the possible association matrices from the possible measurements
  if(num_det != 0) {
    for(uint32_t i = 0; i < _q.rows; ++i) {
      for(uint32_t j = 1; j < _q.cols; ++j) {
        if(_q.at<int32_t>(i, j)) {                              // == 1, associate pair det and track
          tmp_association_matrices.at(hyp_num)(i, 0) = 0;
          tmp_association_matrices.at(hyp_num)(i, j) = 1;
          ++hyp_num;
          if ( j == _q.cols - 1 ) {
              continue;
          }
          for(uint32_t l = 0; l < _q.rows; ++l) {
            if(l != i) {
              for(uint32_t m = j + 1; m < _q.cols; ++m) {// CHECK Q.COLS - 1
                if(_q.at<int32_t>(l, m)) {
                  tmp_association_matrices.at(hyp_num)(i, 0) = 0;
                  tmp_association_matrices.at(hyp_num)(i, j) = 1;
                  tmp_association_matrices.at(hyp_num)(l, 0) = 0;
                  tmp_association_matrices.at(hyp_num)(l, m) = 1;
                  ++hyp_num;
                } //if(q.at<int32_t>(l, m))
              }// m
            } // if l != i
          } // l
        } // if q(i, j) == 1
      } // j
    } // i
  }
  // The last hypothesis is that every detection comes from clutter, all false alarm
  Matrices association_matrices(hyp_num + 1);
  std::copy(tmp_association_matrices.begin(), tmp_association_matrices.begin() + hyp_num + 1, 
	    association_matrices.begin());
  return association_matrices;
}

// Find trackers that are not assocate to any detection 
Tracker::VecBool Tracker::analyze_tracks(const cv::Mat& _q, const std::vector<Detection>& _detections) {
  const cv::Mat& m_q = _q(cv::Rect(1, 0, _q.cols - 1, _q.rows));
  cv::Mat col_sum(cv::Size(m_q.cols, 1), _q.type(), cv::Scalar(0));

  VecBool not_associate(m_q.cols, true); //ALL TRACKS ARE ASSOCIATED
  for(uint32_t i = 0; i < m_q.rows; ++i) {
    col_sum += m_q.row(i);
  }
  cv::Mat nonZero;
  col_sum.convertTo(col_sum, CV_8UC1);

  cv::Mat zero = col_sum == 0;
  cv::Mat zeroValues;
  cv::findNonZero(zero, zeroValues);
  
  for(uint32_t i = 0; i < zeroValues.total(); ++i) {
    not_associate.at(zeroValues.at<cv::Point>(i).x) = false;
  }   
  return not_associate;
}

float Tracker::EuclideanDist(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2) {
    const Eigen::Vector2f& tmp = p1 - p2;
    return sqrt(tmp(0) * tmp(0) + tmp(1) * tmp(1));
}

float Tracker::OpencvMahalanobis(const Eigen::Vector2f &prediction, const Eigen::Vector2f &measurement,
                                 const Eigen::Matrix2f &covariance) {
    cv::Mat covariance_cv;
    cv::eigen2cv(covariance, covariance_cv);
    cv::Mat prediction_cv(cv::Size(2, 1), CV_32FC1);
    cv::Mat measurement_cv(cv::Size(2, 1), CV_32FC1);
    prediction_cv.at<float>(0) = prediction(0);
    prediction_cv.at<float>(1) = prediction(1);
    measurement_cv.at<float>(0) = measurement(0);
    measurement_cv.at<float>(1) = measurement(1);
    return cv::Mahalanobis(prediction_cv, measurement_cv, covariance_cv.inv());
}
