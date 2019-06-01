class myThread (threading.Thread):
   def __init__(self, image, name):
      threading.Thread.__init__(self)
      self.image = image
      self.name = name
   def run(self):
      print(hello)
