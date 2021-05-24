* "Arrival-delay-Final Analysis.ipynb" contains my analysis of clustering ~500K delay events of Translink in 2016, and "Arrival-delay-Clusters-Analysis.pdf" is a chart depicting 'defining features'
of each cluster.


## Delay Events Metadata
<b> duration: </b>The total time in seconds that the delay event lasted

<b> n_stops: </b>The total number of stops comprising the delay event

<b> total_distance: </b>The total distance from point-to-point within the delay event

<b> avg_distance: </b>The average distance from point-to-point within the delay event

<b> avgTravelTime: </b>The average travel time (how long it takes to get from point to point) for each stop within the delay event

<b> avg_ArriveLoad: </b>The average arrive load (how many people were on the bus at the bus stop arrivals) for each stop within the 
delay event

<b> avg_OnsLoad: </b>The average number of people getting on the bus at each stop within the delay event

<b> avg_OffsLoad: </b>The average number of people getting off the bus at each stop within the delay event

<b> avg_DwellTime: </b>The average number of seconds the bus stayed at each stop before moving again

<b> avg_OnAndOffs: </b>The average net difference of people getting on the bus at each stop within the delay event

<b> StartsWithTimingPoint: </b>1 if the delay event started at a timing point. 0 if not

<b> TotalTimingPoints: </b>The total number of timing points that the bus passes within the delay event

<b> Avg_HeadwayOffset: </b>The average number of metres for each stop that the bus was off by according to its headway target.

<b> ProportionBusGappingFlag: </b>The proportion of stops within the delay event that were related to bus gapping

<b> Proportion_HeadwayWithinTarget: </b>The proportion of stops that were within the headway target (far enough away from the bus in front of it by a certain number of metres)
