# CS-02-GenAI-PDF-Table-Extract

Steps:
1. Download the streamlit_deploy.yaml file and create a new stack in AWS cloudformation console.
2. Once the CFN template is successfully running, please login to the EC2 instance created by the CFN
3. Please refer to the Stack output to find the instance id
4. Use SSM to login to the instance and open the EC2 terminal
5. Login as ec2-user and navigate to /home/ce2-user/ directory
6. Run ./start_application.py to run the streamlit application
7. Find the Streamlit URL in the nohup.out and click on the External URL
