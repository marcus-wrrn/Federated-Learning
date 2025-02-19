import logging
import os 

def config_logger(name,log_file,path,level=logging.DEBUG):

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
    return logger
    
def setup_server_logger(path):
    
    if server_logger.hasHandlers():
        return server_logger
    else :
        server_logger = config_logger("server","server.log",path,level=logging.DEBUG)

def setup_client_logger(path):
    if client_logger.hasHandlers():
        return client_logger
    else :
        client_logger = config_logger("client","client.log",path,level=logging.DEBUG)

server_logger = setup_server_logger()
client_logger = setup_client_logger()