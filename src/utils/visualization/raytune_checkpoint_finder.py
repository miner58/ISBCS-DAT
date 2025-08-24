"""
RayTune 실험 결과 분석 도구

이 모듈은 RayTune 실험 결과에서 최적 체크포인트를 찾는 기능을 제공합니다.

주요 기능:
1. max_metric_per_subject.csv에서 실험 정보 추출
2. eeg_tune 폴더에서 해당하는 실험 경로 검색
3. 체크포인트 디렉토리 매칭 및 경로 추출
"""

import pandas as pd
import os
import glob
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class RayTuneCheckpointFinder:
    """
    RayTune 실험 결과에서 체크포인트를 찾는 클래스
    """
    
    def __init__(self, experiment_result_path: str):
        """
        초기화
        
        Args:
            experiment_result_path (str): 실험 결과가 저장된 기본 경로
        """
        self.experiment_result_path = experiment_result_path
        self.analyzing_result_path = os.path.join(experiment_result_path, "analyzing_result")
        self.eeg_tune_path = os.path.join(experiment_result_path, "eeg_tune")
        self.best_checkpoint_path = os.path.join(self.analyzing_result_path, "best_checkpoints.csv")
        
        
        # 경로 존재 확인
        self._validate_paths()
    
    def _validate_paths(self):
        """필요한 경로들이 존재하는지 확인"""
        if not os.path.exists(self.analyzing_result_path):
            raise FileNotFoundError(f"analyzing_result 폴더가 존재하지 않습니다: {self.analyzing_result_path}")
        
        if not os.path.exists(self.eeg_tune_path):
            raise FileNotFoundError(f"eeg_tune 폴더가 존재하지 않습니다: {self.eeg_tune_path}")
        
        csv_path = os.path.join(self.analyzing_result_path, "max_metric_per_subject.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"max_metric_per_subject.csv 파일이 존재하지 않습니다: {csv_path}")

        if not os.path.exists(self.best_checkpoint_path):
            raise FileNotFoundError(f"best_checkpoints.csv 파일이 존재하지 않습니다: {self.best_checkpoint_path}")
        
    
    def step1_extract_csv_data(self) -> pd.DataFrame:
        """
        1단계: max_metric_per_subject.csv 파일에서 필요한 데이터 추출
        
        Returns:
            pd.DataFrame: 추출된 데이터 (test_subject_name, grl_lambda, lnl_lambda)
        """
        csv_path = os.path.join(self.analyzing_result_path, "max_metric_per_subject.csv")
        
        print(f"📄 CSV 파일 읽기: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"✅ CSV 파일 로드 완료. 총 {len(df)}개 행")
        print(f"📋 컬럼 목록: {df.columns.tolist()}")
        
        # 필요한 컬럼 확인
        required_columns = ['test_subject_name', 'grl_lambda', 'lnl_lambda']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ 누락된 컬럼: {missing_columns}")
            print("💡 사용 가능한 컬럼에서 유사한 컬럼을 찾아보겠습니다.")
            
            # 컬럼명 매핑 시도
            column_mapping = {}
            for col in df.columns:
                if 'checkpoint' in col.lower() or 'dir' in col.lower():
                    column_mapping['checkpoint_dir_name'] = col
                elif 'subject' in col.lower() and 'test' not in col.lower():
                    column_mapping['test_subject_name'] = col  
                elif 'grl' in col.lower():
                    column_mapping['grl_lambda'] = col
                elif 'lnl' in col.lower():
                    column_mapping['lnl_lambda'] = col
            
            print(f"🔄 컬럼 매핑: {column_mapping}")
            
            # 컬럼명 변경
            df = df.rename(columns=column_mapping)
        
        # 최종 필요 컬럼 추출
        extracted_columns = []
        for col in required_columns:
            if col in df.columns:
                extracted_columns.append(col)
            else:
                print(f"❌ {col} 컬럼을 찾을 수 없습니다.")
        
        if extracted_columns:
            result_df = df[extracted_columns].copy()
            
            # 데이터 미리보기
            print(f"\n📊 추출된 데이터 미리보기:")
            print(result_df.head())
            
            # 각 컬럼의 고유값 개수
            print(f"\n📈 데이터 통계:")
            for col in extracted_columns:
                unique_count = result_df[col].nunique()
                print(f"  {col}: {unique_count}개의 고유값")
            
            return result_df
        else:
            raise ValueError("필요한 컬럼을 찾을 수 없습니다.")
    
    def step2_find_experiment_paths(self, extracted_data: pd.DataFrame) -> Dict[int, Dict]:
        """
        2단계: eeg_tune 폴더에서 해당하는 실험 경로 찾기
        
        Args:
            extracted_data (pd.DataFrame): 1단계에서 추출한 데이터
            
        Returns:
            Dict[int, Dict]: 각 행별로 매칭된 실험 경로 정보
        """
        print(f"\n🔍 2단계: 실험 경로 검색 시작")
        print(f"📁 검색 경로: {self.eeg_tune_path}")
        
        # eeg_tune 폴더 내의 모든 실험 폴더 찾기
        experiment_folders = []
        for root, dirs, files in os.walk(self.eeg_tune_path):
            for dir_name in dirs:
                if 'TorchTrainer' in dir_name:  # RayTune 실험 폴더 패턴
                    full_path = os.path.join(root, dir_name)
                    experiment_folders.append(full_path)
        
        print(f"📦 총 {len(experiment_folders)}개의 실험 폴더 발견")
        
        # 각 CSV 행에 대해 매칭되는 실험 폴더 찾기
        matched_paths = {}
        
        for index, row in extracted_data.iterrows():
            test_subject_name = row.get('test_subject_name', '')
            grl_lambda = row.get('grl_lambda', '')
            lnl_lambda = row.get('lnl_lambda', '')
            
            print(f"\n🎯 행 {index + 1} 검색 중:")
            print(f"  Subject: {test_subject_name}")
            print(f"  GRL Lambda: {grl_lambda}")  
            print(f"  LNL Lambda: {lnl_lambda}")
            
            # 매칭되는 실험 폴더 찾기
            matched_folder = self._find_matching_experiment_folder(
                experiment_folders, test_subject_name, grl_lambda, lnl_lambda
            )
            
            if matched_folder:
                print(f"  ✅ 매칭 폴더 발견: {os.path.basename(matched_folder)}")
                matched_paths[index] = {
                    'experiment_path': matched_folder,
                    'test_subject_name': test_subject_name,
                    'grl_lambda': grl_lambda,
                    'lnl_lambda': lnl_lambda,
                }
            else:
                print(f"  ❌ 매칭되는 폴더를 찾을 수 없습니다.")
                matched_paths[index] = {
                    'experiment_path': None,
                    'test_subject_name': test_subject_name,
                    'grl_lambda': grl_lambda,
                    'lnl_lambda': lnl_lambda,
                }
        
        print(f"\n📊 매칭 결과: {sum(1 for v in matched_paths.values() if v['experiment_path'] is not None)}/{len(matched_paths)}개 성공")
        
        return matched_paths
    
    def _find_matching_experiment_folder(self, experiment_folders: List[str], 
                                       test_subject_name: str, grl_lambda: float, 
                                       lnl_lambda: float) -> Optional[str]:
        """
        주어진 파라미터와 매칭되는 실험 폴더 찾기
        
        Args:
            experiment_folders: 검색할 실험 폴더 목록
            test_subject_name: 대상 subject 이름
            grl_lambda: GRL lambda 값
            lnl_lambda: LNL lambda 값
            
        Returns:
            매칭되는 폴더 경로 또는 None
        """
        for folder_path in experiment_folders:
            folder_name = os.path.basename(folder_path)
            
            # 폴더명에서 파라미터 추출
            if self._is_folder_match(folder_name, test_subject_name, grl_lambda, lnl_lambda):
                return folder_path
        
        return None
    
    def _is_folder_match(self, folder_name: str, test_subject_name: str, 
                        grl_lambda: float, lnl_lambda: float) -> bool:
        """
        폴더명이 주어진 파라미터와 매칭되는지 확인
        
        Args:
            folder_name: 실험 폴더명
            test_subject_name: 대상 subject 이름  
            grl_lambda: GRL lambda 값
            lnl_lambda: LNL lambda 값
            
        Returns:
            매칭 여부
        """
        # test_subject_name 매칭
        if (test_subject_name and test_subject_name not in folder_name) and 'subject_name' in folder_name:
            return False
        
        # grl_lambda 매칭 (소수점 처리)
        if grl_lambda is not None:
            grl_pattern = f"grl_lambda={grl_lambda:.4f}"
            if grl_pattern not in folder_name:
                # 다른 형식도 시도 (예: 0.001 대신 0.0010)
                grl_pattern_alt = f"grl_lambda={grl_lambda:.3f}"
                if grl_pattern_alt not in folder_name:
                    return False
        
        # lnl_lambda 매칭 (소수점 처리)
        if lnl_lambda is not None:
            lnl_pattern = f"lnl_lambda={lnl_lambda:.4f}"
            if lnl_pattern not in folder_name:
                # 다른 형식도 시도
                lnl_pattern_alt = f"lnl_lambda={lnl_lambda:.3f}"
                if lnl_pattern_alt not in folder_name:
                    return False
        
        return True

    def step3_find_checkpoint_paths(self, matched_paths: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        3단계: 각 실험 경로에서 체크포인트 디렉토리 찾기
        # experiment_path와 matching되는 self.best_checkpoint_path의 path 열의 값을 전달
        
        Args:
            matched_paths: 2단계에서 찾은 실험 경로 정보
            
        Returns:
            최종 체크포인트 경로가 포함된 딕셔너리
        """
        print(f"\n🎯 3단계: 체크포인트 경로 검색 시작")
        
        final_results = {}

        df_best_checkpoint_path = pd.read_csv(self.best_checkpoint_path)
        
        for index, path_info in matched_paths.items():
            experiment_path = path_info['experiment_path']

            # experiment_path와 matching되는 self.best_checkpoint_path의 path 열의 값을 전달
            checkpoint_path = self._matching_best_checkpoint_path_with_experiment_path(experiment_path, df_best_checkpoint_path)
            print(f"  체크포인트 디렉토리 이름: {checkpoint_path}")
            # checkpoint_dir_name가 존재하는 지 폴더인지 확인
            if os.path.isdir(checkpoint_path):
                print(f"  ✅ 체크포인트 디렉토리 이름: {checkpoint_path}")
            else:
                print(f"  ❌ 체크포인트 디렉토리 이름이 유효하지 않습니다: {checkpoint_path}")
                checkpoint_path = None

            print(f"\n📁 행 {index + 1} 체크포인트 검색:")
            print(f"  실험 경로: {experiment_path}")
            print(f"  찾을 체크포인트: {checkpoint_path}")

            final_results[index] = path_info.copy()
            final_results[index]['checkpoint_path'] = checkpoint_path
            
        # 성공률 출력
        successful = sum(1 for v in final_results.values() if v['checkpoint_path'] is not None)
        total = len(final_results)
        print(f"\n📊 체크포인트 검색 결과: {successful}/{total}개 성공 ({successful/total*100:.1f}%)")
        
        return final_results
    
    def _matching_best_checkpoint_path_with_experiment_path(self, experiment_path: str, df_best_checkpoint_path: pd.DataFrame) -> Optional[str]:
        """
        experiment_path와 matching되는 self.best_checkpoint_path의 path 열의 값을 전달
        
        Args:
            experiment_path: 실험 폴더 경로
            df_best_checkpoint_path: best_checkpoint_path DataFrame
            
        Returns:
            체크포인트 디렉토리 이름 또는 None
        """
        # experiment_path와 matching되는 path 열의 값을 찾기
        matching_row = df_best_checkpoint_path[df_best_checkpoint_path['path'].str.contains(experiment_path, na=False)]

        # matching_row가 2개 이상이면 경고 출력
        if len(matching_row) > 1:
            raise ValueError(f"경고: {len(matching_row)}개의 매칭된 행이 발견되었습니다. experiment_path: {experiment_path}")
        
        if not matching_row.empty:
            # 첫 번째 매칭된 행의 path 열 값 반환
            return matching_row.iloc[0]['path']
        
        return None
    
    def create_summary_report(self, final_results: Dict[int, Dict]) -> pd.DataFrame:
        """
        최종 결과 요약 리포트 생성
        
        Args:
            final_results: 3단계 완료 후 결과
            
        Returns:
            요약된 결과 DataFrame
        """
        print(f"\n📋 결과 요약 리포트 생성")
        
        summary_data = []
        
        for index, result in final_results.items():
            summary_data.append({
                'row_index': index + 1,
                'test_subject_name': result['test_subject_name'],
                'grl_lambda': result['grl_lambda'],
                'lnl_lambda': result['lnl_lambda'],
                'experiment_found': result['experiment_path'] is not None,
                'checkpoint_found': result['checkpoint_path'] is not None,
                'experiment_path': result['experiment_path'],
                'checkpoint_path': result['checkpoint_path']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"✅ 요약 완료:")
        print(f"  총 처리된 행: {len(summary_df)}")
        print(f"  실험 경로 발견: {summary_df['experiment_found'].sum()}")
        print(f"  체크포인트 발견: {summary_df['checkpoint_found'].sum()}")
        
        return summary_df
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """
        전체 분석 과정 실행 (1단계 → 2단계 → 3단계)
        
        Returns:
            Tuple[요약 DataFrame, 상세 결과 딕셔너리]
        """
        print("🚀 RayTune 체크포인트 분석 시작\n")
        
        try:
            # 1단계: CSV 데이터 추출
            print("=" * 60)
            print("1️⃣ 1단계: CSV 데이터 추출")
            print("=" * 60)
            extracted_data = self.step1_extract_csv_data()
            
            # 2단계: 실험 경로 찾기  
            print("=" * 60)
            print("2️⃣ 2단계: 실험 경로 검색")
            print("=" * 60)
            matched_paths = self.step2_find_experiment_paths(extracted_data)
            
            # 3단계: 체크포인트 경로 찾기
            print("=" * 60)
            print("3️⃣ 3단계: 체크포인트 경로 검색")
            print("=" * 60)
            final_results = self.step3_find_checkpoint_paths(matched_paths)
            
            # 요약 리포트 생성
            print("=" * 60)
            print("📊 결과 요약")
            print("=" * 60)
            summary_df = self.create_summary_report(final_results)
            
            print(f"\n🎉 분석 완료!")
            
            return summary_df, final_results
            
        except Exception as e:
            print(f"\n❌ 분석 중 오류 발생: {str(e)}")
            raise


def analyze_experiment_results(experiment_result_path: str, save_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    실험 결과 분석을 위한 편의 함수
    
    Args:
        experiment_result_path (str): 실험 결과 폴더 경로
        save_results (bool): 결과를 CSV 파일로 저장할지 여부
        
    Returns:
        Tuple[요약 DataFrame, 상세 결과 딕셔너리]
    """
    finder = RayTuneCheckpointFinder(experiment_result_path)
    summary_df, final_results = finder.run_full_analysis()
    
    if save_results:
        # 결과를 CSV 파일로 저장
        output_path = os.path.join(experiment_result_path, "analyzing_result", "checkpoint_analysis_results.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"\n💾 결과 저장됨: {output_path}")
    
    return summary_df, final_results


if __name__ == "__main__":
    # 예시 실행
    example_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60/1_inner_wireless/ray_results_test1_Day1,8_finetune_ReduceLROnPlateau_LNL_batch16"
    
    try:
        summary_df, results = analyze_experiment_results(example_path)
        print("\n📈 분석 결과 미리보기:")
        print(summary_df.head(10))
    except Exception as e:
        print(f"예시 실행 실패: {e}")
        print("실제 경로를 확인하고 다시 시도하세요.")
