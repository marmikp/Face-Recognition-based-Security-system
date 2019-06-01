import cv2
import datetime
import requests
from db_conn import *
import h5py
import os
import face1
past_time = None
def retrain():
        now = datetime.datetime.now()
        global past_time
        if past_time == None:
                past_time = now
        time_diff =int((now-past_time).total_seconds())/60
        import platform
        if platform.system() == 'Windows':
            win =1
        else:
            win =0

        if 1:#time_diff >1:
                global main_c
                past_time = now
	                # while global_lock.locked():
	                #     continue
	                # global_lock.acquire()
                print('New Face adding...')
                main_c.execute("select * from add_person")
                data = main_c.fetchall()
                main_c.execute("truncate table add_person")
                f = h5py.File('tempFiles/temp.hdf5','a')
                g = h5py.File('tempFiles/embedded2.hdf5','a')
                for d in data:
                        if 1:
                            iemd = f[d[1]].value
                            del f[d[1]] 
                            gg = g.create_group(d[2])
                            print("XXX")
                            if win !=1:
                                os.system("mkdir images/"+str(d[2]))
                                os.system("mv temp_image/"+str(d[1])+" images/"+str(d[2])+"/1.jpg")
                                print("sadas")
                            else:
                                os.system("md images\\"+str(d[2]))
                                os.system("move temp_image\\"+str(d[1])+" images\\"+str(d[2])+"\\1.jpg")

                            no_img=1
                            gg.create_dataset(str(no_img)+".jpg", data=iemd)
                            for key in f.keys():
                                if face1.distance(f[key].value,iemd) < 0.25:
                                    print("in keys")
                                    main_c.execute('delete from notification where  img_name=%s',(key))
                                    no_img=no_img+1
                                    if win !=1:
                                        os.system("mv temp_image/"+str(key)+" images/"+str(d[2])+"/"+str(no_img)+".jpg")
                                    else:
                                        os.system("move temp_image\\"+str(key)+" images\\"+str(d[2])+"\\"+str(no_img)+".jpg")
                                    gg.create_dataset(str(no_img)+".jpg", data=f[key].value)
                                    
                                    del f[key]
                                    
                            while no_img<3:
                                    print(no_img)
                                    nn = no_img+1
                                    if win !=1:
                                        os.system("cp images/"+str(d[2])+"/"+str(no_img)+".jpg images/"+str(d[2])+"/"+str(nn)+".jpg")
                                    else:
                                        os.system("copy images\\"+str(d[2])+"\\"+str(no_img)+".jpg images\\"+str(d[2])+"\\"+str(nn)+".jpg")
                                    no_img = no_img+1
                                    gg.create_dataset(str(no_img)+".jpg", data=iemd)
                            main_c.execute('insert into user_detail(name,no_img,last_update_date) values(%s,%s,%s)',(d[2],no_img,now.strftime("%Y-%m-%d")))
                            db.commit()
                            
                        # except Exception as e:
                        #     print(e)
                # main_c.execute("select * from for_asking where flag = 1")
                # data1 = main_c.fetchall()
                # main_c.execute("delete from for_asking where flag=1")
                # for d1 in data1:
                #         local_c.execute("select no_img from user_detail where name = %s",(d1[1]))
                #         no_img = local_c.fetchone()
                #         no = no_img[0] + 1
                #         sf = len(os.listdir(os.pajoin(path, d1[1])))
                #         s =int(sf)+1
                #         local_c.execute("update user_detail set no_img = %s",(no))
                #         os.system("mv temp_image/"+str(d1[2])+" images/"+str(d1[1])+"/"+s+".jpg")
                f.close()
                g.close()
                #global_lock.release()
                print('### complete face adding ###')                #retrain model
                if not len(data) == 0:
                    pre_load()

                
                #retrain model
                #retrain model
                


################

retrain()

