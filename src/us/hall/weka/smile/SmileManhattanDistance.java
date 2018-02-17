package us.hall.weka.smile;

import java.io.Serializable;

import smile.math.distance.Distance;
import smile.math.distance.ManhattanDistance;
import weka.core.Instances;


/**
 * Weka wrapper class for Smile ManhattanDistance class
 * that will work with Weka ClassDiscovery
 */
public class SmileManhattanDistance implements SmileDistance, Serializable {
	
	Instances data;
	
  /**
   * Constructs a Manhattan Distance object, Instances must be still set.
   */
  public SmileManhattanDistance() {
  }

  /**
   * Constructs an Manhattan Distance object and automatically initializes the
   * ranges.
   * 
   * @param data 	the instances the distance function should work on
   */
  public SmileManhattanDistance(Instances data) {
    this.data = data;
  }	
  
  public Distance getSmileDistance() {
  	return new ManhattanDistance();
  }
}