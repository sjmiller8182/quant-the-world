# Predicting Location via Indoor Positioning Systems

## Summary

Real Time Location Systems (RTLS) that are capable of tracking business assets and people in special
circumstances, are a popular area of research. Warehouse distribution and delivery services have grown
more rapidly due to the COVID-19 pandemic, and with that growth, we can expect the relative importance
of tracking assets at scale to increase. In this case study, we assess whether it is possible to predict locations
based on the measurement of WiFi signals from fixed access points (AP)1
. Additionally, two APs are placed
in close proximity, we assess the impact of using both of these APs. The location sensing system analyzed
in this case study is located in an indoor office environment on a single floor of a building

[See Full Report](./real-time_location_system_case_study.pdf)

**Language**: R

## Data

Data was provided by the instructor (link provided below).
Use `get_data.R` to download the data for this case study.

```bash
$ Rscript get_data.R
```

## Provided Materials

* Starter code: [http://rdatasciencecases.org/GeoLoc/code.R](http://rdatasciencecases.org/GeoLoc/code.R)
* Data
  * offline_data: [http://rdatasciencecases.org/Data/offline.final.trace.txt](http://rdatasciencecases.org/Data/offline.final.trace.txt)
  * online_data: [http://rdatasciencecases.org/Data/online.final.trace.txt](http://rdatasciencecases.org/Data/online.final.trace.txt)
