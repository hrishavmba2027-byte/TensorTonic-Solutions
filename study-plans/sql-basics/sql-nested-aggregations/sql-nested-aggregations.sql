with cte as(
    select order_date, sum(amount) as daily_revenue,
    count(id) as daily_count
    from orders
    group by order_date
)
select round(avg(daily_count),2) as avg_daily_orders, 
round(avg(daily_revenue),2) as avg_daily_revenue,
max(daily_count) as busiest_day_orders
from cte