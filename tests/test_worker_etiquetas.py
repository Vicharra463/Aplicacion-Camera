import worker


def test_etiqueta_roboflow():
    assert worker.etiqueta_fuente(
        [("roboflow", "ABC-123", 0.9)]) == "_api-Roboflow"


def test_etiqueta_plate_recognizer():
    assert worker.etiqueta_fuente(
        [("api", "ABC-123", 0.9)]) == "_api-PlateRecognizer"


def test_etiqueta_motor_local_sin_prefijo():
    assert worker.etiqueta_fuente(
        [("crnn", "ABC-123", 0.9)]) == ""
    assert worker.etiqueta_fuente(
        [("rapidocr", "ABC-123", 0.6)]) == ""


def test_roboflow_tiene_prioridad_sobre_api():
    assert worker.etiqueta_fuente(
        [("api", "ABC-123", 0.9), ("roboflow", "ABC-123", 0.8)]
    ) == "_api-Roboflow"


def test_lista_vacia_o_none():
    assert worker.etiqueta_fuente([]) == ""
    assert worker.etiqueta_fuente(None) == ""


def test_etiqueta_no_contamina_sufijo_de_pasada():
    et = worker.etiqueta_fuente([("roboflow", "ABC-123", 0.9)])
    assert et.startswith("_")
    assert " " not in et


def test_texto_de_api_acepta_placa():
    assert worker.texto_de_api("AWI-935") == "AWI-935"
    assert worker.texto_de_api("awi-935") == "AWI-935"


def test_texto_de_api_normaliza_letras_confundidas():
    # S en la ultima posicion es 5 (mismo criterio que ocr._normalizar)
    assert worker.texto_de_api("AWI-93S") == "AWI-935"


def test_texto_de_api_rechaza_rotulos_que_no_son_placa():
    # Visto en produccion 2026-10-31: Roboflow leyo rotulos y entraron
    # a la BD como si fueran placas.
    assert worker.texto_de_api("PERU") == ""
    assert worker.texto_de_api("HERO") == ""
    assert worker.texto_de_api("F.BRUTO: 2.110\nP.NETO: 1.110") == ""
    assert worker.texto_de_api("LIMA") == ""
    assert worker.texto_de_api("VISTONY") == ""
    assert worker.texto_de_api("```MARKDOWN\n\n```") == ""


def test_texto_de_api_extrae_la_placa_de_texto_multilinea():
    # plate_text real del workflow: la placa viene con rótulos al lado
    # (sticker PERU, marca, textos del vehiculo).
    assert worker.texto_de_api("PERU\nT2K-790") == "T2K-790"
    assert worker.texto_de_api("FEB 94\nH1G-954") == "H1G-954"
    assert worker.texto_de_api("PERU\nF2K-343") == "F2K-343"
    # _normalizar canoniza a AAA-999: UN-3077 (2-4) sale como UN3-077
    assert worker.texto_de_api("100\nUN-3077") == "UN3-077"


def test_texto_de_api_marcas_que_simulan_placa_son_ambiguas():
    # Verificado contra ocr._normalizar: TOYOTA -> TOY-0TA (O->0 en
    # pos 4) y MEXICO -> MEX-1C0 (I->1, O->0) tienen FORMA de placa.
    # Con dos candidatas no se adivina: decide la cadena local sobre
    # la misma bbox (consenso CRNN + PlateRecognizer).
    assert worker.texto_de_api("D5W-782\nTOYOTA") == ""
    assert worker.texto_de_api(
        "MEXICO\nPERU\nB7Q-409\nREPUBLICA DEL PERU") == ""


def test_texto_de_api_ambiguo_rechaza_y_deja_a_la_cadena_local():
    # Dos sub-cadenas con forma de placa: no se adivina, decide
    # PlateRecognizer/motor local sobre la misma bbox.
    assert worker.texto_de_api("PERU\nCOY-876\nUN-3077") == ""
    assert worker.texto_de_api("UN-1910\nPERU\nBLX-739") == ""


def test_texto_de_api_vacio_o_sin_motor():
    assert worker.texto_de_api("") == ""
    assert worker.texto_de_api(None) == ""


def test_texto_de_api_exige_exactamente_6_alfanumericos():
    # La regla central del sistema: una placa son EXACTAMENTE 6
    # caracteres alfanumericos (3 letras + 3 digitos). Todo lo que
    # no cumple no es placa, venga de donde venga.
    assert worker.texto_de_api("ABC-123") == "ABC-123"      # 6: pasa
    assert worker.texto_de_api("ABC-12") == ""              # 5: rechaza
    assert worker.texto_de_api("ABC-1234") == ""            # 7: rechaza
    assert worker.texto_de_api("ABC12345") == ""            # 8: rechaza
    assert worker.texto_de_api("123456") == ""              # sin letras
    assert worker.texto_de_api("AAAAAA") == ""              # sin digitos
    # OJO: "ABCDEF" si pasa: D en posicion 4 se lee 0 (misma regla
    # de confusion de _normalizar) -> "ABC-0EF".


def test_color_caja_roboflow_verde():
    assert worker.color_caja("_api-Roboflow") == (0, 255, 0)


def test_color_caja_plate_recognizer_ambar():
    assert worker.color_caja("_api-PlateRecognizer") == (0, 255, 255)


def test_color_caja_motor_local_azul():
    assert worker.color_caja("") == (255, 0, 0)
    assert worker.color_caja("_2da-pasada") == (255, 0, 0)


def test_color_caja_sufijo_de_pasada_no_pierde_la_fuente():
    assert worker.color_caja("_api-Roboflow_2da-pasada") == (0, 255, 0)
    assert worker.color_caja(
        "_api-PlateRecognizer_3ra-pasada") == (0, 255, 255)
