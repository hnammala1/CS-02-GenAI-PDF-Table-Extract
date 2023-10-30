# CS-02-GenAI-PDF-Table-Extract

Steps:
1. Download the streamlit_deploy.yaml file and create a new stack in AWS cloudformation console. Below python libraries should have been installed as a prerequisite
    * boto3
    * streamlit
    * streamlit_chat
    * langchain
    * amazon-textract-response-parser
    * amazon-textract-prettyprinter
    * amazon-textract-helper and
    * faiss-cpu
    * opensearch-py
    * requests_aws4auth

  # PDF Tables Extraction
  # Install the libraries with the following code:
  pip install boto3 streamlit streamlit_chat langchain amazon-textract-response-parser amazon-textract-prettyprinter amazon-textract-helper faiss-cpu opensearch-py requests_aws4auth

3. Please make sure to use your PC's public IP address in the CIDR input parameter (e.g. 73.13.100.24/32) 
4. Once the CFN template is successfully running, please login to the EC2 instance created by the CFN
5. Please refer to the Stack output to find the instance id
6. Use SSM to login to the instance and open the EC2 terminal
7. Login as root and check the /var/log/cloud-init-output.log to see if the initialization scripts have successfully run
8. Login as ec2-user and navigate to /home/ce2-user/ directory
9. Run ./start_application.sh to run the streamlit application
10. Find the Streamlit URL in the nohup.out and click on the External URL
