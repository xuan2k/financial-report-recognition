class LackRequestData(Exception):
    def __init__(self, message="Not enough request data"):
        self.type = "LackRequestData"
        self.message = message
        super().__init__(self.message)

class Unauthorized(Exception):
    def __init__(self, message="Token is missing"):
        self.type = "Unauthorized"
        self.message = message
        super().__init__(self.message)

class AccessForbidden(Exception):
    def __init__(self, message="User forbidden to access the content"):
        self.type = "AccessForbidden"
        self.message = message
        super().__init__(self.message)