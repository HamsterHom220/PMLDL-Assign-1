from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['admin@example.org'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    'execution_timeout': timedelta(days=1),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

dag = DAG(
    'a1-dag',
    default_args=default_args,
    description='Assign1 DAG',
    schedule_interval=timedelta(hours=2),
)

t1 = BashOperator(
    task_id='preprocess_data',
    bash_command='python ~/MLOps/Assign1/code/datasets/preprocess.py',
    dag=dag,
)

t2 = BashOperator(
    task_id='train_model',
    bash_command='python ~/MLOps/Assign1/code/models/train.py',
    dag=dag,
)

t3 = BashOperator(
    task_id='deploy_app',
    bash_command='cd ~/MLOps/Assign1/ && docker compose up',
    dag=dag,
)

t1 >> t2 >> t3