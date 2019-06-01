import pymysql as PyMySQL
import ftplib
# Open database connection
db = PyMySQL.connect("localhost","root","","main_db" )
db1 = PyMySQL.connect("localhost","root","","local_db" )
main_c = db.cursor()
local_c=db1.cursor()
server_url = 'http://localhost'
# c= db1.cursor()
# c.execute("INSERT INTO temp(activity,time,flag) SELECT * FROM (SELECT 'jaimin','15:02',0) AS tp WHERE NOT EXISTS (SELECT * from temp where activity='jaimin')")
# db1.commit()

db_live = PyMySQL.connect("localhost","root","","main_db" )
main_live = db_live.cursor()