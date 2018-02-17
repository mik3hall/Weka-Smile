package us.hall.weka.smile;

import java.io.Serializable;

import smile.math.distance.Distance;
import smile.math.distance.MinkowskiDistance;
import weka.core.Instances;


/**
 * Weka wrapper class for Smile MinkowskiDistance class
 * that will work with Weka ClassDiscovery
 */
public class SmileMinkowskiDistance implements SmileDistance, Serializable {
	
	Instances data;
	
  /**
   * Constructs an Minkowski Distance object, Instances must be still set.
   */
  public SmileMinkowskiDistance() {
  }

  /**
   * Constructs an Euclidean Distance object and automatically initializes the
   * ranges.
   * 
   * @param data 	the instances the distance function should work on
   */
  public SmileMinkowskiDistance(Instances data) {
    this.data = data;
  }	
  
  public Distance getSmileDistance() {
  	return new MinkowskiDistance(1);
  }
}