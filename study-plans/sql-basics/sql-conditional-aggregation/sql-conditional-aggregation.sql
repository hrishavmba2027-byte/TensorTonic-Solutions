-- Write your SQL query here
select department,
count(id) as total_tickets,
sum(case when status = 'open' then 1 else 0 end) as open_count,
(select count(id) from tickets t1 where t1.department = t2.department and status = 'in_progress') as in_progress_count,
sum(case when status = 'closed' then 1 else 0 end) as closed_count
from tickets t2
group by department
order by total_tickets desc, department asc