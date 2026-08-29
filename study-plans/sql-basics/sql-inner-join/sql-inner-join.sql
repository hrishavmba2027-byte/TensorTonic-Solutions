-- Write your SQL query here
select name, salary, dept_name
from employees emp 
inner join departments dept 
on emp.dept_id = dept.id 
order by name asc