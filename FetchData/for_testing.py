INSERT INTO test_data
		(`id`, `question_body`, `python`, `javascript`, `java`, `c`, `r`, `mysql`, `html`, `if_statement`, `while_loop`, `for_loop`, `css`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
		WHERE NOT EXISTS (
    		SELECT id FROM complete_train_data2 WHERE id = id;
		) 


insert into test_data 
	(id, question_body, python, javascript, java, c, r, mysql, html, if_statement, while_loop, for_loop, css) values (10075304, 'hello', 1,0,0,0,0,0,0,0,1,0,0) 

	select t1.id from complete_train_data2 
	where not exists (select id from test_data t2 where t2.id = t1.id)

where not exists (select id from complete_train_data2 where id = id);


INSERT INTO AdminAccounts 
    (Name)
SELECT t1.name
  FROM Matrix t1
 WHERE NOT EXISTS(SELECT id
                    FROM AdminAccounts t2
                   WHERE t2.Name = t1.Name)


DELETE FROM test_data
    USING test_data, complete_train_data2
    WHERE test_data.id = complete_train_data2.id