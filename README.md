Weka/Smile
==========

Weka package to interface to [Smile](http://haifengl.github.io/smile/). 

Supported so far. 

Classification
==============

KNN 
===
Is an implementation of the Smile KNN classifier. A very slight modification of the Weka
Ibk class to call Smile KNN methods at the appropriate places. 

Clustering
==========

DBScan 
======

Density based classifier. Includes Weka wrapper classes for the Smile distance 
algorithm implementations. This seemed necessary to work correcty with Weka 
ClassDiscovery.

Example: Based on a Smile example. 
In the data directory select the file clustering/chameleon/out_t4.8k.arff

```
weka.clusterers.DBScan -M 20 -R 10.0 -A us.hall.weka.smile.SmileEuclideanDistance -S 1
```

Smile handles clustering somewhat differently from Weka in having an 'outlier' result.
Initially I was handling this by setting outliers to the 0 cluster. DBScan currently 
attempts to add an additional cluster for outliers.

You can visualize the results by using the appropriate meta click on the DBScan entry
in the result list. For OS X this is 'option + click'. Change the visualization to make
X: attr1 and Y: attr2.

BIRCH
=====
Indicated as a good algorithm for large datasets. 

Example: Again based on a Smile example.
In the data directory select the file clustering/gaussian/six.arff

```
weka.clusterers.BIRCH -k 6 -B 5 -T 0.5 -S 1
```

This still uses cluster number 0 for outliers. It can be visualized in the same way.

SpectralClustering
==================
Dimension reducing clustering based on similarity. 

Example: From a Smile demo.
In the data directory select the file clustering/nonconvex/sincos.arff

```
weka.clusterers.SpectralClustering -k 2 -w 0.25 -S 1
```

For some reason -w width of .1 works with Smile but not with this.

You can compare the visualizations against the demos in Smile. They seem roughly correct.

