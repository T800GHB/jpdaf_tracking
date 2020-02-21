#include "global_tracker.h"

using namespace JPDAFTracker;

GlobalTracker::GlobalTracker(const TrackerParam& _param)
  : Tracker(_param) {
  trackID_ = 0;
  init_ = true;
  startTracking_ = false;
  rng_ = cv::RNG(12345);
}

void GlobalTracker::track(const GlobalTracker::Detections& _detections) {
  if(init_) {
    prev_detections_.clear();
    for(const auto& det : _detections) {
      prev_detections_.emplace_back(Eigen::Vector2f(det.x(), det.y()));
    }
    init_ = false;
  } else if((not init_) and (not startTracking_)) {
    not_associated_.clear();
    for(const auto& det : _detections) {
      not_associated_.emplace_back(Eigen::Vector2f(det.x(), det.y()));
    }
    manage_new_tracks();
    if(localTrackers_.size() > 0) {
      tracks_.clear();
      tracks_ = localTrackers_.at(0)->tracks();
      startTracking_ = true;
    } else {
      prev_detections_ = not_associated_;
    }
  } else {
    not_associated_.clear();
    VecBool isAssoc(_detections.size(), false); //all the detections are not associated

    for(const auto& tracker : localTrackers_) {
      tracker->track(_detections, isAssoc, trackID_);
    }

    for (auto i = 0; i < isAssoc.size(); ++i) {
        if (not isAssoc.at(i)) {
            not_associated_.emplace_back(Eigen::Vector2f(_detections.at(i).x(), _detections.at(i).y()));
        }
    }

    delete_tracks();
    
    tracks_.clear();
    for(const auto& tracker : localTrackers_) {
      Tracks tr = tracker->tracks();
      for(const auto& t : tr) {
	    tracks_.emplace_back(t);
      }
    }
    //Create a q matrix with a width = clutter + number of tracks
    cv::Mat_<int32_t> q(cv::Size(tracks_.size() + 1, _detections.size()), int32_t(0));
    
    std::vector<Eigen::Vector2f> selected_detections;
  
    //ASSOCIATION
    std::vector<bool> not_associate;
    associate(selected_detections, q, _detections);
    
    if(q.total() > 0) {
      //ASSIGN ALL THE NOT ASSOCIATED TRACKS
      not_associate = analyze_tracks(q, _detections);
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
        }
      }
    }

    manage_new_tracks(); 
  }
}

void GlobalTracker::delete_tracks() {
  for(int32_t i = localTrackers_.size() - 1; i >= 0; --i) {
    if(localTrackers_.at(i)->size() == 0) {
      localTrackers_.erase(localTrackers_.begin() + i);
    }
  }
}

void GlobalTracker::manage_new_tracks() {
  const uint32_t& prevDetSize = prev_detections_.size();
  const uint32_t& deteSize = not_associated_.size();
  if(prevDetSize == 0) {
    prev_detections_ = not_associated_;
  } else if (deteSize == 0) {
    prev_detections_.clear();
  } else {
    cv::Mat assigments_bin = cv::Mat::zeros(cv::Size(deteSize, prevDetSize), CV_32SC1);
    cv::Mat costMat = cv::Mat(cv::Size(deteSize, prevDetSize), CV_32FC1);

    assignments_t assignments;
    std::vector<float> costs(deteSize * prevDetSize);

    for(uint32_t i = 0; i < prevDetSize; ++i) {
      for(uint32_t j = 0; j < deteSize; ++j) {
	    costs.at(i + j * prevDetSize ) = EuclideanDist(not_associated_.at(j), prev_detections_.at(i));
	    costMat.at<float>(i, j) = costs.at(i + j * prevDetSize );
      }
    }
    
    	  
    AssignmentProblemSolver APS;
    APS.Solve(costs, prevDetSize, deteSize, assignments, AssignmentProblemSolver::optimal);

    const uint32_t& assSize = assignments.size();
    
    for(uint32_t i = 0; i < assSize; ++i) {
      if(assignments[i] != -1 and costMat.at<float>(i, assignments[i]) < param_.assocCost) {
	    assigments_bin.at<int32_t>(i, assignments[i]) = 1;
      }
    }
    
    const uint32_t& rows = assigments_bin.rows;
    const uint32_t& cols = assigments_bin.cols;
    
    LocalTracker_ptr tracker = LocalTracker_ptr(new LocalTracker(param_));
        
    
    for(uint32_t i = 0; i < rows; ++i) {
      for(uint32_t j = 0; j < cols; ++j) {
        if(assigments_bin.at<int32_t>(i, j)) {
          const float& vx = not_associated_.at(j).x() - prev_detections_.at(i).x();
          const float& vy = not_associated_.at(j).y() - prev_detections_.at(i).y();
          std::shared_ptr<Track> tr(new Track(param_.dt, param_.target_delta, not_associated_.at(j).x(),
                  not_associated_.at(j).y(), vx, vy, param_.g_sigma, param_.gamma, param_.R ));
          tracker->push_back(tr);
        }
      }
    }

    if(tracker->size() > 0){
      localTrackers_.emplace_back(tracker);
    }

    cv::Mat notAssignedDet(cv::Size(assigments_bin.cols, 1), CV_32SC1, cv::Scalar(0));
    for(uint32_t i = 0; i < assigments_bin.rows; ++i) {
      notAssignedDet += assigments_bin.row(i);
    }
    
    notAssignedDet.convertTo(notAssignedDet, CV_8UC1);
    notAssignedDet = notAssignedDet == 0;
    
    cv::Mat dets;
    cv::findNonZero(notAssignedDet, dets);
    prev_detections_.clear();
    for(uint32_t i = 0; i < dets.total(); ++i) {
        const uint32_t& idx = dets.at<cv::Point>(i).x;
        prev_detections_.emplace_back(not_associated_.at(idx));
    }
  }
}


void GlobalTracker::associate(std::vector< Eigen::Vector2f >& _selected_detections, cv::Mat& _q, 
			const std::vector< Detection >& _detections) {
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
      if(mah <= param_.global_g_sigma && eucl <= param_.global_assocCost) {
        _q.at<int32_t>(num_selected_det, 0) = 1;
        _q.at<int32_t>(num_selected_det, i) = 1;
        found = true;
      }
      ++i;
    }

    if(found) {
      _selected_detections.emplace_back(det);
      num_selected_det++;
    }
    ++k;
  }
  _q = _q(cv::Rect(0, 0, tracks_.size() + 1, num_selected_det));
}