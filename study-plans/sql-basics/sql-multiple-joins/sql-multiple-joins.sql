-- Write your SQL query here
select username, experiment_name, variant, revenue
from users u 
inner join experiment_assignments e
on u.id = e.user_id
inner join conversions c
on u.id = c.user_id
order by experiment_name asc, revenue desc, username asc