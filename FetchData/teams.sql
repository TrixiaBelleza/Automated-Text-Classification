-- drop database if exists questions_db;
-- create database questions_db;
-- use questions_db;

-- create table questions (
-- 	id VARCHAR(50),
-- 	question_body VARCHAR(65000),
-- 	date_passed DATE,
-- 	status VARCHAR(50),
-- 	legal_history VARCHAR(50),
-- 	date_filed DATE,
-- 	constraint bill_number_pk primary key(bill_number)
-- );
drop database if exists team_db;
create database team_db;
use team_db;

CREATE TABLE team (
  team_id INT PRIMARY KEY,
  team_name VARCHAR(50)
);
INSERT INTO team (team_id, team_name) VALUES (1, 'Dwarfs');

CREATE TABLE team_members (
  team_id INT,
  member_name VARCHAR(20),
  CONSTRAINT `fk_team_id`
		FOREIGN KEY (team_id) REFERENCES team (team_id)
);

INSERT INTO team_members VALUES 
  (1, 'Sleepy'),
  (1, 'Dopey'),
  (1, 'Sneezy'),
  (1, 'Happy'),
  (1, 'Grumpy'),
  (1, 'Doc'),
  (1, 'Bashful');