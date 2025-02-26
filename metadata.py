"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="SmolVLM2 Frame Classifier",
        description="Applies SmolVLM2-2.2B-Instruct multimodal model to video frames for classification.",
        app_license="Apache 2.0",
        identifier="smolvlm2-classifier",
        url="https://github.com/clamsproject/app-smolvlm2-classifier"
    )

    # and then add I/O specifications: an app must have at least one input and one output
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_output(AnnotationTypes.TimePoint)
    metadata.add_output(DocumentTypes.TextDocument)
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(
        name='frameInterval', type='integer', default=30,
        description='The interval at which to extract frames from the video for classification. '
        'Default is every 30 frames.'
    )
    metadata.add_parameter(
        name='classifierPrompt', type='string', default='Classify what is shown in this video frame. Output only the primary category that best describes the content.',
        description='Prompt to use for classification of each frame.'
    )
    
    # add parameter for config file name
    metadata.add_parameter(
        name='config', type='string', default="config/default.yaml", description='Name of the config file to use.'
    )
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
