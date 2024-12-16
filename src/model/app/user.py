
class User:
    
    def __init__(self, username, user_id):
        self.username=username
        self.user_id=user_id
        
    def getUsername(self):
        return self.username
    
    def getId(self):
        return self.user_id
        