"""
파일 불러오기 모듈
- CSV : 쉼표(,) / 세미콜론(;) / 탭 구분자, 소수점 쉼표(0,5), 한글 인코딩 자동 처리
- Excel : 시트 선택, 제목(헤더) 행 위치 선택
- 텍스트(.txt/.data) : 공백 구분
글자로 읽힌 숫자 열(단위 행, 천 단위 쉼표, 소수점 쉼표 등)은 자동으로 숫자로 변환한다.
"""
import re
import pandas as pd

_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def _to_number(series):
    """글자 열 하나를 숫자로 변환 (천 단위 쉼표와 소수점 쉼표를 구분)"""
    text = series.astype(str).str.strip().str.replace("\u00a0", "", regex=False)
    has_comma = text.str.contains(",", regex=False)
    if has_comma.any() and text[has_comma].str.match(_THOUSANDS).all():
        text = text.str.replace(",", "", regex=False)          # 1,234.5 → 1234.5
    else:
        text = text.str.replace(",", ".", regex=False)         # 0,125 → 0.125
    return pd.to_numeric(text, errors="coerce")


def coerce_numeric(df, min_ratio=0.8):
    """값의 80% 이상이 숫자인 글자 열을 숫자형으로 바꾼다. 빈 행·열은 제거."""
    df = df.dropna(how="all").dropna(axis=1, how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        conv = _to_number(df[col])
        if conv.notna().mean() >= min_ratio:
            df[col] = conv
    return df


def numeric_columns(df):
    """X/Y축으로 쓸 수 있는 숫자형 열 목록"""
    return df.select_dtypes(include="number").columns.tolist()


def excel_sheets(file):
    """Excel 파일의 시트 이름 목록"""
    if hasattr(file, "seek"):
        file.seek(0)
    return pd.ExcelFile(file).sheet_names


def _read_csv(file):
    candidates = []
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                df = pd.read_csv(file, sep=sep, encoding=enc, dtype=str)
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            df = coerce_numeric(df)
            candidates.append((len(numeric_columns(df)), df.shape[1], df))
        if candidates:
            break
    if not candidates:
        raise ValueError("CSV 파일 형식을 인식하지 못했습니다.")
    return max(candidates, key=lambda c: (c[0], c[1]))[2]   # 숫자 열이 가장 많이 잡힌 방식


def load_file(file, sheet=0, header_row=0):
    """파일을 DataFrame으로 읽는다. header_row: 열 이름이 있는 행 (0부터)"""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return _read_csv(file)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        if hasattr(file, "seek"):
            file.seek(0)
        return coerce_numeric(pd.read_excel(file, sheet_name=sheet, header=header_row))
    if name.endswith((".txt", ".data")):
        df = pd.read_csv(file, sep=r"\s+", header=None)
        df.columns = [f"열{i + 1}" for i in range(df.shape[1])]
        return coerce_numeric(df)
    raise ValueError("CSV, Excel(.xlsx/.xls), 텍스트(.txt/.data) 파일만 지원합니다.")


def clean_xy(df, x_col, y_col):
    """선택한 두 열에서 빈칸을 제거하고 x 기준으로 정렬"""
    sub = df[[x_col, y_col]].dropna().sort_values(x_col)
    return sub[x_col].to_numpy(float), sub[y_col].to_numpy(float)
