from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleWrapper:
    def __init__(self):
        # authentificate
        self.__api = KaggleApi()
        self.__api.authenticate()
    
    def submit_result(self, csv_path, competition_id, description=None):
        self.__api.competition_submit(csv_path, description, competition_id)
