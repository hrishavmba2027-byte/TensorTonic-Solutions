-- Write your SQL query here
select segment_name, metric_name
from segments s
cross join metrics m
order by segment_name asc, metric_name asc