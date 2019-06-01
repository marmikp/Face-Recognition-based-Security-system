import datetime
import cv2
import ftplib
from db_conn import *
import os
import time
import threading
import json
import h5py
import requests
filter_time = None
filter_array = []
def filter(name):
        global filter_array
        global filter_time
        th =threading.current_thread()
        now = datetime.datetime.now()
        dt =now.strftime("%Y-%m-%d")
        time  =now.strftime("%H:%M")
        cstr  = '{"activity":"'+name+'","time":"'+time+'","date":"'+dt+'"}'
        if filter_time == None:
                filter_time = now
        time_diff =int((now-filter_time).total_seconds())/60
        if time_diff > 10:
                filter_array = []
                filter_time = now
        if cstr not in filter_array:
                filter_array.append(cstr)
                print("diffrent #######")
                return False
        else:
                print("repeat #######")
                return True;

      
        # try:

        #         th.local_c.execute("INSERT INTO temp(date1,activity,flag) VALUES(%s,%s,0)",(dt,str(json.loads(cstr))))

        #         return False
        # except Exception as e:
        #         return True
                
                        
                        

def notification(now,msg,iname):
        th =threading.current_thread()
        dt =now.strftime("%Y-%m-%d")
        time  =now.strftime("%H:%M")
        th.main_c.execute("insert into notification(msg,date,time,img_name,flag) values(%s,%s,%s,%s,0)",(msg,dt,time,iname))
        
def update_log(now,msg,mode):
        th =threading.current_thread()
        dt =now.strftime("%Y-%m-%d")
        time  =now.strftime("%H:%M")
        th.main_c.execute("insert into log_data(date,time,activity,mode) values(%s,%s,%s,%s)",(str(dt),str(time),msg,mode))
        
def upload_img(now,imu,emd):
        th =threading.current_thread()
        img_name = str(now.strftime("%Y%m%d_%H%M%S"))+".jpg"
        cv2.imwrite("temp_image/"+img_name,imu)
        cv2.imwrite("temp_image/thumbnail/"+img_name,cv2.resize(imu.copy(),(120,120)))
        file = {'image': open("temp_image/"+img_name,'rb')}                  # file to send
        r= requests.post(server_url+'/receive_file.php', files=file,data={'location': '/'})
        file = {'image': open("temp_image/thumbnail/"+img_name,'rb')}                  # file to send
        r= requests.post(server_url+'/receive_file.php', files=file,data={'location': '/thumbnail'})
        th.img_n= img_name
        th.temd=emd

        
def data_maintain(nam, im,emd):
        th =threading.current_thread()
        sper =th.per
        if sper == "person":
                pe=1
        else:
                pe=0
        
        if nam =="noface":                      #no_face
                fn=0
        elif nam != "unknown":
                fn = 2                                  #known face
        else:                                           
                fn = 1                                  #unknown_face

        
        
        now = datetime.datetime.now()

        
        th.main_c.execute("select low,normal,high from mode where id=1")
        mode_d =th.main_c.fetchone()


        img_name = str(now.strftime("%Y%m%d_%H%M%S"))+".jpg"
                        
        if mode_d[0] == 1:
                if filter(nam+sper+"/low"):
                        return None                                                                    #low
                update_log(now,nam+sper,"low")
        elif mode_d[1] == 1:
                if filter(nam+sper+"/normal"):
                        return None                                                                    #normal 
                update_log(now,nam+sper,"normal")
                if fn==0 and pe==1:
                        notification(now,"person detected",img_name)
                        upload_img(now,im,emd)
                if fn==1:
                        notification(now,"unknown face detected",img_name)
                        upload_img(now,im,emd )                     
        else:
                if filter(nam+sper+"/high"):
                        return None                                                                     #high
                update_log(now,nam+sper,"high")
                if pe==1 or fn!=0:                      ######### alert ###### mqtt ######
                        notification(now,nam+sper+" detected",img_name)
                        upload_img(now,im,emd)



      
        
                
# def for_asking(nam, im,emd=None):
#         if filter("ask "+nam):
#                 return None
#         th =threading.current_thread()
#         now = datetime.datetime.now()
#         th.local_c.execute("select no_img from user_detail where name = %s",nam)
#         no_img = th.local_c.fetchone()
#         number = no_img[0]
#         print("handle for asking called")
#         if number < 9:
#                         img_name = str(now.isoformat())+".jpg"
#                         upload_img(now,im,emd)
#                         th.main_c.execute("insert into for_asking(name,img_name,flag) values(%s,%s,0)",(nam,img_name))
#         else:
#                 data_maintain("unknown", im)
        
