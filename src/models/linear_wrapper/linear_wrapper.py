from src.models.linear_wrapper.linear_model import LinearModel


class LinearWrapper():
    
    def __init__(self, linear_model=None):
        """Initialization.
        
        Args:
            linear_model (LinearModel): Linear Algorithm
        """
        self.linear_model = linear_model
    
    def save(self, linear_folder):
        """Save the linear object to a folder

        Args:
            linear_folder (str): The saving folder name
        """
        self.linear_model.save(linear_folder)
        
    @classmethod
    def load(cls, linear_folder):
        """Load linear model

        Args:
            linear_folder (str): The folder to load

        Returns:
            cls: An instance of LinearWrapper
        """
        linear_model = LinearModel.load(linear_folder)
        return cls(linear_model)