#include "local_tracker.h"

using namespace JPDAFTracker;

LocalTracker::LocalTracker(const TrackerParam& _param)
  : Tracker(_param) {
}


void LocalTracker::track(const std::vector< Detection >& _detections, VecBool& _isAssoc, uint32_t& _trackID) {
  for(auto& track : tracks_) {
    track->predict();
    if(track->getId() == -1 && track->isAlive() && track->getEntropy() == Track::TrackState::ACCEPT) {
      track->setId(_trackID++);
      track->setColor(cv::Scalar(rng_.uniform(0, 255), rng_.uniform(0, 255), rng_.uniform(0, 255)));
    }
  }
  //Create a q matrix with a width = clutter + number of tracks
  cv::Mat_<int32_t> q(cv::Size(tracks_.size() + 1, _detections.size()), int32_t(0));
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
    const Matrices& association_matrices = generate_hypothesis(q);
      
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
  for(int32_t i = tracks_.size() - 1; i >= 0; --i) {
    if(!tracks_.at(i)->isAlive() && tracks_.at(i)->getId() != -1) {
      tracks_.erase(tracks_.begin() + i);
    }
  }
}

void LocalTracker::associate(std::vector< Eigen::Vector2f >& _selected_detections, cv::Mat& _q, 
			const std::vector<Detection>& _detections, VecBool& _isAssoc) {
  //Extracting the measurements inside the validation gate for all the tracks
  uint32_t num_selected_det = 0;
  uint32_t k = 0;
  
  for(const auto& detection : _detections) {
    Eigen::Vector2f det;
    det << detection.x(), detection.y();
    uint32_t i = 1;
    bool found = false;
    for(const auto& track : tracks_) {

      const Eigen::Vector2f& tr = track->getLastPredictionEigen();
      const Eigen::Matrix2f& S = track->S();
      const float& mah = OpencvMahalanobis(tr, det, S);
      const float& eucl = EuclideanDist(det, tr);

      if(mah <= param_.g_sigma && eucl <= param_.assocCost) {
        _q.at<int32_t>(num_selected_det, 0) = 1;
        _q.at<int32_t>(num_selected_det, i) = 1;
        found = true;
      }
      ++i;
    }
    if(found) {
      _selected_detections.emplace_back(det);
      _isAssoc.at(k) = true;
      num_selected_det++;
    }
    ++k;
  }
  _q = _q(cv::Rect(0, 0, tracks_.size() + 1, num_selected_det));
}