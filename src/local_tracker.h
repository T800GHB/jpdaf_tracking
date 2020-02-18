#ifndef _LOCAL_TRACKER_
#define _LOCAL_TRACKER_

#include "tracker.h"

namespace JPDAFTracker
{
  class LocalTracker : public Tracker
  {
    public:
      LocalTracker(const TrackerParam& _param);
      void track(const Detections& _detections, VecBool& _isAssoc, uint& _trackID);
      inline void push_back(const Track_ptr& _track) {
	    tracks_.push_back(_track);
      }

    private:
     void associate(Vec2f& _selected_detections, cv::Mat& _q, const Detections& _detections, VecBool& _isAssoc);
     void delete_tracks();
  };
}

#endif