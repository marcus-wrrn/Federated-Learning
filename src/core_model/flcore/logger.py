import logging
import os 

def setup_logger(name,log_file,path,level=logging.DEBUG):

    ## SETUP LOGGER 
    logger = logging.getLogger(name)
    logger.setLevel(level)


    ## LOGGER HANDLERS 
    log_path = os.path.join(path,log_file)
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler

    ## LOGGER SETUP

    file_handler.setLevel(level)
    console_handler.setLevel(level)

    ## FORMAT MESSAGES
    file_format = logging.Formatter("%(asctime)s -  %(name)s - %(levelname)s - %(message)s")
    file_debug_format = logging.Formatter("%(asctime)s - %(name)s - %(pathname)s - %(funcName)s - %(lineno)s : %(message)s")
    

    if(level == logging.DEBUG):
        file_handler.setFormatter(file_debug_format)
    else:
        file_handler.setFormatter(file_format)

    ## ADD HANDLERS TO LOGGER 
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
server_logger = setup_logger("server","server.log",path=os.getcwd(),level=logging.DEBUG)
client_logger = setup_logger("client","client.log",path = os.getcwd(),level=logging.DEBUG)