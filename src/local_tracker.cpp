#include "local_tracker.h"

using namespace JPDAFTracker;

LocalTracker::LocalTracker(const TrackerParam& _param)
  : Tracker(_param) {
}


void LocalTracker::track(const std::vector< Detection >& _detections, VecBool& _isAssoc, uint& _trackID) {
  for(auto& track : tracks_) {
    track->predict();
    if(track->getId() == -1 && track->isAlive() && track->getEntropy() == Track::TrackState::ACCEPT) {
      track->setId(_trackID++);
      track->setColor(cv::Scalar(rng_.uniform(0, 255), rng_.uniform(0, 255), rng_.uniform(0, 255)));
    }
  }
  //Create a q matrix with a width = clutter + number of tracks
  cv::Mat_<int> q(cv::Size(tracks_.size() + 1, _detections.size()), int(0));
  std::vector<Eigen::Vector2f> selected_detections;
  
  //ASSOCIATION
  std::vector<bool> not_associate;
  associate(selected_detections, q, _detections, _isAssoc);
  
  //NO ASSOCIATIONS
  if(q.total() == 0) {
    for(const auto& track : tracks_) {
      track->notDetected();
    }
  } else {
    //CHECK ASSOCIATIONS
    not_associate = analyze_tracks(q, _detections); //ASSIGN ALL THE NOT ASSOCIATED TRACKS
      
    //HYPOTHESIS
    const Matrices& association_matrices = generate_hypothesis(selected_detections, q);
      
    //COMPUTE JOINT PROBABILITY
    beta_ = joint_probability(association_matrices, selected_detections);
    last_beta_ = beta_.row(beta_.rows() - 1);
      

    for(auto i = 0; i < tracks_.size(); ++i) {
        const auto& track = tracks_.at(i);
        if(not_associate.at(i)) {
            //KALMAN PREDICT STEP
            track->gainUpdate(last_beta_(i));
            //UPDATE AND CORRECT
            track->update(selected_detections, beta_.col(i), last_beta_(i));
        } else {
            track->notDetected();
        }
    }

  }
  delete_tracks();
}

void LocalTracker::delete_tracks() {
  for(int i = tracks_.size() - 1; i >= 0; --i) {
    if(!tracks_.at(i)->isAlive() && tracks_.at(i)->getId() != -1) {
      tracks_.erase(tracks_.begin() + i);
    }
  }
}

void LocalTracker::associate(std::vector< Eigen::Vector2f >& _selected_detections, cv::Mat& _q, 
			const std::vector< Detection >& _detections, VecBool& _isAssoc) {
  //Extracting the measurements inside the validation gate for all the tracks
  uint validationIdx = 0;
  uint j = 0;
  
  for(const auto& detection : _detections) {
    Eigen::Vector2f det;
    det << detection.x(), detection.y();
    uint i = 1;
    bool found = false;
    cv::Mat det_cv(cv::Size(2, 1), CV_32FC1);
    det_cv.at<float>(0) = det(0);
    det_cv.at<float>(1) = det(1);
    for(auto& track : tracks_) {

      const Eigen::Vector2f& tr = track->getLastPredictionEigen();
      cv::Mat tr_cv(cv::Size(2, 1), CV_32FC1);
      tr_cv.at<float>(0) = tr(0);
      tr_cv.at<float>(1) = tr(1);

      const Eigen::Matrix2f& S = track->S().inverse();
      cv::Mat S_cv;
      cv::eigen2cv(S, S_cv);

      const float& mah = cv::Mahalanobis(tr_cv, det_cv, S_cv);
      const float& eucl = EuclideanDist(det, tr);
      if(mah <= param_.g_sigma && eucl <= param_.assocCost) {
        _q.at<int>(validationIdx, 0) = 1;
        _q.at<int>(validationIdx, i) = 1;
        found = true;
      }
      ++i;
    }
    if(found) {
      _selected_detections.emplace_back(det);
      _isAssoc.at(j) = true;
      validationIdx++;
    }
    ++j;
  }
  _q = _q(cv::Rect(0, 0, tracks_.size() + 1, validationIdx));
}