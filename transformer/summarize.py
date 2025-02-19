from pathlib import Path
from config import get_config, latest_weights_file_path
from train import get_model
from tokenizers import Tokenizer
import torch
from pyvi import ViTokenizer


def summarize(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    sentence = ViTokenizer.tokenize(sentence)
    tokenizer = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["language"])))
    )
    print(str(Path(config["tokenizer_file"].format(config["language"]))))
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    print("ok1")
    model.load_state_dict(state["model_state_dict"])
    print("ok")
    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer.encode(sentence).ids
        if len(source) > config["src_len"] - 2:
            source = source[: config["src_len"] - 2]
        source = torch.cat(
            [
                torch.tensor([tokenizer.token_to_id("<s>")], dtype=torch.int64),
                torch.tensor(source, dtype=torch.int64),
                torch.tensor([tokenizer.token_to_id("</s>")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer.token_to_id("<pad>")]
                    * (config["src_len"] - len(source) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)
        source = source.unsqueeze(0)
        source_mask = (
            (source != tokenizer.token_to_id("<pad>"))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )
        print(source.shape)
        print(source_mask.shape)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = (
            torch.empty(1, 1)
            .fill_(tokenizer.token_to_id("</s>"))
            .type_as(source)
            .to(device)
        )

        # Print the source sentence and target start prompt

        # Generate the translation word by word
        while decoder_input.size(1) < config["tgt_len"]:
            # build mask for target and calculate output
            decoder_mask = (
                (
                    torch.triu(
                        torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                        diagonal=1,
                    )
                    == 0
                )
                .type(torch.int)
                .type_as(source_mask)
                .to(device)
            )
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(device),
                ],
                dim=1,
            )

            # print the translated word
            print(f"{tokenizer.decode([next_word.item()])}", end=" ")

            # break if we predict the end of sentence token
            if next_word == tokenizer.token_to_id("</s>"):
                break

    # convert ids to tokens
    return tokenizer.decode(decoder_input[0].tolist())


# with open("test.txt", "r", encoding="utf-8") as f:
#     text = f.read()
# text = text.strip()
# text = text.replace("\n", " ")
# text = ViTokenizer.tokenize(text)
# # read sentence from argument
# print(text)
text = """Tổng_thống Nga Vladimir_Putin_phát_biểu tại Diễn_đàn_Quốc_tế về Bắc_Cực ở St . Petersburg hôm_nay . " Ngay từ đầu chúng_tôi đã nói rằng đội_ngũ của Mueller sẽ không tìm thấy bằng_chứng , bởi chúng_tôi biết rõ hơn ai hết là Nga không can_thiệp bất_cứ cuộc bầu_cử nào của Mỹ " , Tổng_thống_Vladimir_Putin hôm_nay phát_biểu , đề_cập tới cuộc điều_tra của công_tố_viên đặc_biệt Robert_Mueller . Putin nhấn_mạnh kết_quả không nằm ngoài dự_đoán , đồng_thời gọi cuộc điều_tra của Mueller là " đầu_voi_đuôi_chuột " . " Sự thông_đồng giữa Tổng_thống_Donald_Trump và Nga , điều mà Mueller đang tìm_kiếm , không hề tồn_tại " , Tổng_thống_Nga nói . Ông Mueller cuối tháng trước gửi báo_cáo_mật cho Bộ_trưởng Tư_pháp_Mỹ_William_Barr về kết_quả điều_tra cáo_buộc Nga thông_đồng với chiến_dịch tranh_cử của Trump năm 2016 , trong đó kết_luận Tổng_thống Mỹ không có hành_vi phạm_tội . Bộ_trưởng Barr hôm_nay cho_biết bản báo_cáo của Mueller sẽ được công_bố vào tuần tới nhưng chỉ tiết_lộ một phần nhằm giới_hạn những chi_tiết có_thể ảnh_hưởng đến các cuộc điều_tra đang diễn ra , tài_liệu_mật hoặc thông_tin ảnh_hưởng tới bên thứ ba . Đảng Dân_chủ muốn ông Barr công_khai toàn_bộ báo_cáo nhưng đề_nghị này bị từ_chối . Tổng_thống_Trump nhiều lần chỉ_trích cuộc điều_tra của Mueller là " cuộc săn_phù_thuỷ " và cho_rằng đây là nỗ_lực hạ_bệ bất_thành . Nga cũng kiên_quyết phủ_nhận cáo_buộc thông_đồng với chiến_dịch tranh_cử của Trump , đồng_thời yêu_cầu truyền_thông Mỹ_xin_lỗi vì đưa tin sai_lệch về Moskva ."""
summarize(text)
