-- Write your SQL query here
select month, revenue, 
lag(revenue,1,0) over (order by month asc) as prev_revenue,
(revenue - prev_revenue) as revenue_change
from monthly_revenue
order by month asc